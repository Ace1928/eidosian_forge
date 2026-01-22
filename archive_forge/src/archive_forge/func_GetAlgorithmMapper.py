from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetAlgorithmMapper(api_version=None):
    messages = apis.GetMessagesModule(api_version)
    algorithm_enum = messages.PkixPublicKey.SignatureAlgorithmValueValuesEnum
    return arg_utils.ChoiceEnumMapper('algorithm_enum', algorithm_enum, include_filter=lambda name: 'UNSPECIFIED' not in name)