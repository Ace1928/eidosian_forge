import copy
import logging
from openstack import exceptions as sdk_exceptions
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
def _get_volume_availability_zones(self, parsed_args):
    volume_client = self.app.client_manager.sdk_connection.volume
    data = []
    try:
        data = list(volume_client.availability_zones())
    except Exception as e:
        LOG.debug('Volume availability zone exception: %s', e)
        if parsed_args.volume:
            message = _('Availability zones list not supported by Block Storage API')
            LOG.warning(message)
    result = []
    for zone in data:
        result += _xform_volume_availability_zone(zone)
    return result