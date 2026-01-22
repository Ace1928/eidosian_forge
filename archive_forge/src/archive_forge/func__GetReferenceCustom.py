from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def _GetReferenceCustom(self):
    """Tries to resolve the reference for runtime: custom.

    If the user has an app.yaml with runtime: custom we will look in the root
    of their source directory for a custom build pipeline named cloudbuild.yaml.

    This should only be called if there is *not* a Dockerfile in the source
    root since that means they just want to build and deploy that Docker image.

    Returns:
      BuilderReference or None
    """
    if self.runtime == 'custom':
        log.debug('Using local cloud build file [%s] for custom runtime.', Resolver.CLOUDBUILD_FILE)
        return BuilderReference(self.runtime, _Join('file:///' + self.source_dir.replace('\\', '/').strip('/'), Resolver.CLOUDBUILD_FILE))
    return None