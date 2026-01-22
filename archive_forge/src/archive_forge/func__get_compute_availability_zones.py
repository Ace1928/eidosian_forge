import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _get_compute_availability_zones(self, parsed_args):
    compute_client = self.app.client_manager.sdk_connection.compute
    try:
        data = list(compute_client.availability_zones(details=True))
    except sdk_exceptions.ForbiddenException:
        try:
            data = compute_client.availability_zones(details=False)
        except Exception:
            raise
    result = []
    for zone in data:
        result += _xform_compute_availability_zone(zone, parsed_args.long)
    return result