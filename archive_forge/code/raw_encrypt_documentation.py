from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
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
Verifies integrity fields in RawEncryptResponse.

    Note: This methods assumes that self._PerformIntegrityVerification() is True
    and that all request CRC32C fields were pupolated.
    Args:
      req:
        messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsRawEncryptRequest()
        object
      resp: messages.RawEncryptResponse() object.

    Returns:
      Void.
    Raises:
      e2e_integrity.ServerSideIntegrityVerificationError if the server reports
      request integrity verification error.
      e2e_integrity.ClientSideIntegrityVerificationError if response integrity
      verification fails.
    