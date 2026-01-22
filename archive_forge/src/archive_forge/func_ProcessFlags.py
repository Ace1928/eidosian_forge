from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
def ProcessFlags(self, args):
    fields_to_update = []
    if args.external_key_uri:
        fields_to_update.append('externalProtectionLevelOptions.externalKeyUri')
    if args.ekm_connection_key_path:
        fields_to_update.append('externalProtectionLevelOptions.ekmConnectionKeyPath')
    if args.state:
        fields_to_update.append('state')
    if not fields_to_update:
        raise kms_exceptions.UpdateError('An error occurred: --external-key-uri or --ekm-connection-key-path or --state must be specified.')
    return fields_to_update