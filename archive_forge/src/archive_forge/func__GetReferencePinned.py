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
def _GetReferencePinned(self):
    """Tries to resolve the reference for when a runtime is pinned.

    Usually a runtime is looked up in the manifest and resolved to a
    configuration file. The user does have the option of 'pinning' their build
    to a specific configuration by specifying the absolute path to a builder
    in the runtime field.

    Returns:
      BuilderReference or None
    """
    if self.runtime.startswith('gs://'):
        log.debug('Using pinned cloud build file [%s].', self.runtime)
        return BuilderReference(self.runtime, self.runtime)
    return None