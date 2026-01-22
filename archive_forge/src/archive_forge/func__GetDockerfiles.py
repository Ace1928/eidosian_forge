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
def _GetDockerfiles(info, dockerfile_dir):
    """Returns map of in-memory Docker-related files to be packaged.

  Returns the files in-memory, so that we don't have to drop them on disk;
  instead, we include them in the archive sent to App Engine directly.

  Args:
    info: (googlecloudsdk.api_lib.app.yaml_parsing.ServiceYamlInfo)
      The service config.
    dockerfile_dir: str, path to the directory to fingerprint and generate
      Dockerfiles for.

  Raises:
    UnsatisfiedRequirementsError: Raised if the code in the directory doesn't
      satisfy the requirements of the specified runtime type.

  Returns:
    A dictionary of filename relative to the archive root (str) to file contents
    (str).
  """
    params = ext_runtime.Params(appinfo=info.parsed, deploy=True)
    configurator = fingerprinter.IdentifyDirectory(dockerfile_dir, params)
    if configurator:
        dockerfiles = configurator.GenerateConfigData()
        return {d.filename: d.contents for d in dockerfiles}
    else:
        raise UnsatisfiedRequirementsError('Your application does not satisfy all of the requirements for a runtime of type [{0}].  Please correct the errors and try again.'.format(info.runtime))