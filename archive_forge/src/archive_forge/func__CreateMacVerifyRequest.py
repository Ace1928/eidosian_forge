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
def _CreateMacVerifyRequest(self, args):
    try:
        data = self._ReadFileOrStdin(args.input_file, max_bytes=65536)
    except EnvironmentError as e:
        raise exceptions.BadFileException('Failed to read input file [{0}]: {1}'.format(args.input_file, e))
    try:
        mac = self._ReadFileOrStdin(args.signature_file, max_bytes=64)
    except EnvironmentError as e:
        raise exceptions.BadFileException('Failed to read input file [{0}]: {1}'.format(args.input_file, e))
    messages = cloudkms_base.GetMessagesModule()
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacVerifyRequest(name=flags.ParseCryptoKeyVersionName(args).RelativeName())
    if self._PerformIntegrityVerification(args):
        data_crc32c = crc32c.Crc32c(data)
        mac_crc32c = crc32c.Crc32c(mac)
        req.macVerifyRequest = messages.MacVerifyRequest(data=data, mac=mac, dataCrc32c=data_crc32c, macCrc32c=mac_crc32c)
    else:
        req.macVerifyRequest = messages.MacVerifyRequest(data=data, mac=mac)
    return req