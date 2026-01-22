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
def _CreateAsymmetricSignRequestOnDigest(self, args):
    try:
        digest = get_digest.GetDigest(args.digest_algorithm, args.input_file)
    except EnvironmentError as e:
        raise exceptions.BadFileException('Failed to read input file [{0}]: {1}'.format(args.input_file, e))
    messages = cloudkms_base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsAsymmetricSignRequest(name=flags.ParseCryptoKeyVersionName(args).RelativeName())
    if self._PerformIntegrityVerification(args):
        digest_crc32c = crc32c.Crc32c(getattr(digest, args.digest_algorithm))
        req.asymmetricSignRequest = messages.AsymmetricSignRequest(digest=digest, digestCrc32c=digest_crc32c)
    else:
        req.asymmetricSignRequest = messages.AsymmetricSignRequest(digest=digest)
    return req