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
def _GetReferenceFromManifest(self):
    """Tries to resolve the reference by looking up the runtime in the manifest.

    Calculate the location of the manifest based on the builder root and load
    that data. Then try to resolve a reference based on the contents of the
    manifest.

    Returns:
      BuilderReference or None
    """
    manifest_file_name = Resolver.BUILDPACKS_MANIFEST_NAME if self.use_flex_with_buildpacks else Resolver.MANIFEST_NAME
    manifest_uri = _Join(self.build_file_root, manifest_file_name)
    log.debug('Using manifest_uri [%s]', manifest_uri)
    try:
        manifest = Manifest.LoadFromURI(manifest_uri)
        return manifest.GetBuilderReference(self.runtime)
    except FileReadError:
        log.debug('', exc_info=True)
        return None