from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.api_lib.firestore import databases
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.firestore import flags
from googlecloudsdk.core import properties
def DatabaseCmekConfig(self, args):
    if args.kms_key_name is not None:
        return api_utils.GetMessages().GoogleFirestoreAdminV1CmekConfig(kmsKeyName=args.kms_key_name)
    return api_utils.GetMessages().GoogleFirestoreAdminV1CmekConfig()