import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalVolumeTarget(command.Command):
    """Unregister baremetal volume target(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('volume_targets', metavar='<volume target>', nargs='+', help=_('UUID(s) of the volume target(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for volume_target in parsed_args.volume_targets:
            try:
                baremetal_client.volume_target.delete(volume_target)
                print(_('Deleted volume target %s') % volume_target)
            except exc.ClientException as e:
                failures.append(_('Failed to delete volume target %(volume_target)s: %(error)s') % {'volume_target': volume_target, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))