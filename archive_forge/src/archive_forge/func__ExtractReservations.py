from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bms.bms_client import BmsClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.bms import flags
from googlecloudsdk.command_lib.bms import util
def _ExtractReservations(network):
    """Extracts reservations from network object."""
    out = []
    for res in network.reservations:
        reservation_dict = {}
        reservation_dict['name'] = network.name
        reservation_dict['id'] = network.id
        reservation_dict['start_address'] = res.startAddress
        reservation_dict['end_address'] = res.endAddress
        reservation_dict['note'] = res.note
        out.append(reservation_dict)
    return out