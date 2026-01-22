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
def GetBuilderReference(self):
    """Resolve the builder reference.

    Returns:
      BuilderReference, The reference to the builder configuration.

    Raises:
      BuilderResolveError: if this fails to resolve a builder.
    """
    builder_def = self._GetReferenceCustom() or self._GetReferencePinned() or self._GetReferenceFromManifest() or self._GetReferenceFromLegacy()
    if not builder_def:
        raise BuilderResolveError('Unable to resolve a builder for runtime: [{runtime}]'.format(runtime=self.runtime))
    return builder_def