from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import environments as env_util
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def GetReservationAffinityEnum():
    type_enum = None
    if args.IsSpecified('reservation_affinity'):
        reservation_affinity_message = messages.ReservationAffinity
        type_enum = arg_utils.ChoiceEnumMapper(arg_name='reservation-affinity', message_enum=reservation_affinity_message.ConsumeReservationTypeValueValuesEnum, include_filter=lambda x: 'UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(args.reservation_affinity))
    return type_enum