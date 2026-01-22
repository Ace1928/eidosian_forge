import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalVolumeTarget(command.ShowOne):
    """Show baremetal volume target details."""
    log = logging.getLogger(__name__ + '.ShowBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('volume_target', metavar='<id>', help=_('UUID of the volume target.'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.VOLUME_TARGET_DETAILED_RESOURCE.fields, help=_('One or more volume target fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        volume_target = baremetal_client.volume_target.get(parsed_args.volume_target, fields=fields)._info
        volume_target.pop('links', None)
        return zip(*sorted(volume_target.items()))