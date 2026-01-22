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
from googlecloudsdk.command_lib.kms import get_digest
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _CreateAsymmetricSignRequest(self, args):
    if self._SignOnDigest(args):
        return self._CreateAsymmetricSignRequestOnDigest(args)
    else:
        return self._CreateAsymmetricSignRequestOnData(args)