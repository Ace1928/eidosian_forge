from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
from apitools.base.py import exceptions as apitools_exceptions
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import build as app_build
from googlecloudsdk.api_lib.app import cloud_build
from googlecloudsdk.api_lib.app import docker_image
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import runtime_builders
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.api_lib.app.runtimes import fingerprinter
from googlecloudsdk.api_lib.cloudbuild import build as cloudbuild_build
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions as s_exceptions
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import exceptions as api_lib_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.tools import context_util
import six
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
def _NeedsDockerfile(info, source_dir):
    """Returns True if the given directory needs a Dockerfile for this app.

  A Dockerfile is necessary when there is no Dockerfile in source_dir,
  regardless of whether we generate it here on the client-side, or in Cloud
  Container Builder server-side.

  The reason this function is more complicated than that is that it additionally
  verifies the sanity of the provided configuration by raising an exception if:

  - The runtime is "custom", but no Dockerfile is present
  - The runtime is not "custom", and a Dockerfile or cloudbuild.yaml is present
  - The runtime is "custom", and has both a cloudbuild.yaml and a Dockerfile.

  (The reason cloudbuild.yaml is tied into this method is that its use should be
  mutually exclusive with the Dockerfile.)

  Args:
    info: (googlecloudsdk.api_lib.app.yaml_parsing.ServiceYamlInfo). The
      configuration for the service.
    source_dir: str, the path to the service's source directory

  Raises:
    CloudbuildYamlError: if a cloudbuild.yaml is present, but the runtime is not
      "custom".
    DockerfileError: if a Dockerfile is present, but the runtime is not
      "custom".
    NoDockerfileError: Raised if a user didn't supply a Dockerfile and chose a
      custom runtime.
    CustomRuntimeFilesError: if a custom runtime had both a Dockerfile and a
      cloudbuild.yaml file.

  Returns:
    bool, whether Dockerfile generation is necessary.
  """
    has_dockerfile = os.path.exists(os.path.join(source_dir, config.DOCKERFILE))
    has_cloudbuild = os.path.exists(os.path.join(source_dir, runtime_builders.Resolver.CLOUDBUILD_FILE))
    if info.runtime == 'custom':
        if has_dockerfile and has_cloudbuild:
            raise CustomRuntimeFilesError('A custom runtime must have exactly one of [{}] and [{}] in the source directory; [{}] contains both'.format(config.DOCKERFILE, runtime_builders.Resolver.CLOUDBUILD_FILE, source_dir))
        elif has_dockerfile:
            log.info('Using %s found in %s', config.DOCKERFILE, source_dir)
            return False
        elif has_cloudbuild:
            log.info('Not using %s because cloudbuild.yaml was found instead.', config.DOCKERFILE)
            return True
        else:
            raise NoDockerfileError('You must provide your own Dockerfile when using a custom runtime. Otherwise provide a "runtime" field with one of the supported runtimes.')
    else:
        if has_dockerfile:
            raise DockerfileError('There is a Dockerfile in the current directory, and the runtime field in {0} is currently set to [runtime: {1}]. To use your Dockerfile to build a custom runtime, set the runtime field to [runtime: custom]. To continue using the [{1}] runtime, please remove the Dockerfile from this directory.'.format(info.file, info.runtime))
        elif has_cloudbuild:
            raise CloudbuildYamlError('There is a cloudbuild.yaml in the current directory, and the runtime field in {0} is currently set to [runtime: {1}]. To use your cloudbuild.yaml to build a custom runtime, set the runtime field to [runtime: custom]. To continue using the [{1}] runtime, please remove the cloudbuild.yaml from this directory.'.format(info.file, info.runtime))
        log.info('Need Dockerfile to be generated for runtime %s', info.runtime)
        return True