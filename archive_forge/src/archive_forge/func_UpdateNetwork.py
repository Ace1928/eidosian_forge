from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions as apilib_exceptions
from googlecloudsdk.command_lib.bms import util
import six
def UpdateNetwork(self, network_resource, labels, ip_reservations):
    """Update an existing network resource."""
    updated_fields = []
    ip_reservations_messages = []
    if labels is not None:
        updated_fields.append('labels')
    if ip_reservations is not None:
        updated_fields.append('reservations')
        for ip_reservation in ip_reservations:
            ip_reservations_messages.append(self.messages.NetworkAddressReservation(startAddress=ip_reservation.start_address, endAddress=ip_reservation.end_address, note=ip_reservation.note))
    network_msg = self.messages.Network(name=network_resource.RelativeName(), labels=labels, reservations=ip_reservations_messages)
    request = self.messages.BaremetalsolutionProjectsLocationsNetworksPatchRequest(name=network_resource.RelativeName(), network=network_msg, updateMask=','.join(updated_fields))
    return self.networks_service.Patch(request)