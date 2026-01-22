import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalVolumeTarget(command.ShowOne):
    """Create a new baremetal volume target."""
    log = logging.getLogger(__name__ + '.CreateBaremetalVolumeTarget')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalVolumeTarget, self).get_parser(prog_name)
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', required=True, help=_('UUID of the node that this volume target belongs to.'))
        parser.add_argument('--type', dest='volume_type', metavar='<volume type>', required=True, help=_("Type of the volume target, e.g. 'iscsi', 'fibre_channel'."))
        parser.add_argument('--property', dest='properties', metavar='<key=value>', action='append', help=_('Key/value property related to the type of this volume target. Can be specified multiple times.'))
        parser.add_argument('--boot-index', dest='boot_index', metavar='<boot index>', type=int, required=True, help=_('Boot index of the volume target.'))
        parser.add_argument('--volume-id', dest='volume_id', metavar='<volume id>', required=True, help=_('ID of the volume associated with this target.'))
        parser.add_argument('--uuid', dest='uuid', metavar='<uuid>', help=_('UUID of the volume target.'))
        parser.add_argument('--extra', dest='extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)' % parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if parsed_args.boot_index < 0:
            raise exc.CommandError(_('Expected non-negative --boot-index, got %s') % parsed_args.boot_index)
        field_list = ['extra', 'volume_type', 'properties', 'boot_index', 'node_uuid', 'volume_id', 'uuid']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        fields = utils.args_array_to_dict(fields, 'properties')
        fields = utils.args_array_to_dict(fields, 'extra')
        volume_target = baremetal_client.volume_target.create(**fields)
        data = dict([(f, getattr(volume_target, f, '')) for f in res_fields.VOLUME_TARGET_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)