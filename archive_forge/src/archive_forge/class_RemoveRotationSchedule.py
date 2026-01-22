from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import resource_args
class RemoveRotationSchedule(base.UpdateCommand):
    """Remove the rotation schedule for a key.

  Removes the rotation schedule for the given key.

  ## EXAMPLES

  The following command removes the rotation schedule for the key
  named `frodo` within the keyring `fellowship` and location `global`:

    $ {command} frodo \\
        --location=global \\
        --keyring=fellowship
  """

    @staticmethod
    def Args(parser):
        resource_args.AddKmsKeyResourceArgForKMS(parser, True, 'key')

    def Run(self, args):
        client = cloudkms_base.GetClientInstance()
        messages = cloudkms_base.GetMessagesModule()
        crypto_key_ref = flags.ParseCryptoKeyName(args)
        req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest(name=crypto_key_ref.RelativeName(), cryptoKey=messages.CryptoKey(), updateMask='rotationPeriod,nextRotationTime')
        return client.projects_locations_keyRings_cryptoKeys.Patch(req)