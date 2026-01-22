import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalAllocation(command.ShowOne):
    """Create a new baremetal allocation."""
    log = logging.getLogger(__name__ + '.CreateBaremetalAllocation')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalAllocation, self).get_parser(prog_name)
        parser.add_argument('--resource-class', dest='resource_class', help=_('Resource class to request.'))
        parser.add_argument('--trait', action='append', dest='traits', help=_('A trait to request. Can be specified multiple times.'))
        parser.add_argument('--candidate-node', action='append', dest='candidate_nodes', help=_('A candidate node for this allocation. Can be specified multiple times. If at least one is specified, only the provided candidate nodes are considered for the allocation.'))
        parser.add_argument('--name', dest='name', help=_('Unique name of the allocation.'))
        parser.add_argument('--uuid', dest='uuid', help=_('UUID of the allocation.'))
        parser.add_argument('--owner', dest='owner', help=_('Owner of the allocation.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        parser.add_argument('--wait', type=int, dest='wait_timeout', default=None, metavar='<time-out>', const=0, nargs='?', help=_('Wait for the new allocation to become active. An error is returned if allocation fails and --wait is used. Optionally takes a timeout value (in seconds). The default value is 0, meaning it will wait indefinitely.'))
        parser.add_argument('--node', help=_('Backfill this allocation from the provided node that has already been deployed. Bypasses the normal allocation process.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        if not parsed_args.node and (not parsed_args.resource_class):
            raise exc.ClientException(_('--resource-class is required except when --node is used'))
        field_list = ['name', 'uuid', 'extra', 'resource_class', 'traits', 'candidate_nodes', 'node', 'owner']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        fields = utils.args_array_to_dict(fields, 'extra')
        allocation = baremetal_client.allocation.create(**fields)
        if parsed_args.wait_timeout is not None:
            allocation = baremetal_client.allocation.wait(allocation.uuid, timeout=parsed_args.wait_timeout)
        data = dict([(f, getattr(allocation, f, '')) for f in res_fields.ALLOCATION_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)