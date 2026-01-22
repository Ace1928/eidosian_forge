import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalVolumeConnector(command.Command):
    """Unregister baremetal volume connector(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalVolumeConnector')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalVolumeConnector, self).get_parser(prog_name)
        parser.add_argument('volume_connectors', metavar='<volume connector>', nargs='+', help=_('UUID(s) of the volume connector(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for volume_connector in parsed_args.volume_connectors:
            try:
                baremetal_client.volume_connector.delete(volume_connector)
                print(_('Deleted volume connector %s') % volume_connector)
            except exc.ClientException as e:
                failures.append(_('Failed to delete volume connector %(volume_connector)s: %(error)s') % {'volume_connector': volume_connector, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))