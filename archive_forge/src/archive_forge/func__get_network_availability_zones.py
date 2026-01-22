import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _get_network_availability_zones(self, parsed_args):
    network_client = self.app.client_manager.network
    try:
        network_client.find_extension('Availability Zone', ignore_missing=False)
    except Exception as e:
        LOG.debug('Network availability zone exception: ', e)
        if parsed_args.network:
            message = _('Availability zones list not supported by Network API')
            LOG.warning(message)
        return []
    result = []
    for zone in network_client.availability_zones():
        result += _xform_network_availability_zone(zone)
    return result