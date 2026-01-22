from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.kms import crc32c
from googlecloudsdk.command_lib.kms import e2e_integrity
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _MaybeStripResourceVersion(self, req_name, resp_name):
    """Strips the trailing '/cryptoKeyVersions/xx' from Response's resource name.

    If request's resource name is a key and not a version, returns response's
    resource name with the trailing '/cryptoKeyVersions/xx' suffix stripped.
    Args:
      req_name: String.
        CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest.name.
      resp_name: String. EncryptResponse.name.

    Returns:
      resp_resource_name with '/cryptoKeyVersions/xx' suffix stripped.
    """
    if req_name.find('/cryptoKeyVersions/') == -1:
        return resp_name.partition('/cryptoKeyVersions/')[0]
    else:
        return resp_name