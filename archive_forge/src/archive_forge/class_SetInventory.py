import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
class SetInventory(command.Lister, version.CheckerMixin):
    """Replaces the set of inventory records for the resource provider.

    Note that by default this is a full replacement of the existing inventory.
    If you want to retain the existing inventory and add a new resource class
    inventory, you must specify all resource class inventory, old and new, or
    specify the ``--amend`` option.

    If a specific inventory field is not specified for a given resource class,
    it is assumed to be the total, i.e. ``--resource VCPU=16`` is equivalent to
    ``--resource VCPU:total=16``.

    Example::

        openstack resource provider inventory set <uuid>             --resource VCPU=16             --resource MEMORY_MB=2048             --resource MEMORY_MB:step_size=128
    """

    def get_parser(self, prog_name):
        parser = super(SetInventory, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider or UUID of the aggregate, if --aggregate is specified')
        fields_help = '\n'.join(('{} - {}'.format(f, INVENTORY_FIELDS[f]['help'].lower()) for f in INVENTORY_FIELDS))
        parser.add_argument('--resource', metavar='<resource_class>:<inventory_field>=<value>', help='String describing resource.\n' + RC_HELP + '\n<inventory_field> (optional) can be:\n' + fields_help, default=[], action='append')
        parser.add_argument('--aggregate', action='store_true', help='If this option is specified, the inventories for all resource providers that are members of the aggregate will be set. This option requires at least ``--os-placement-api-version 1.3``')
        parser.add_argument('--amend', action='store_true', help='If this option is specified, the inventories will be amended instead of being fully replaced')
        parser.add_argument('--dry-run', action='store_true', help='If this option is specified, the inventories that would be set will be returned without actually setting any inventories')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        if parsed_args.aggregate:
            self.check_version(version.ge('1.3'))
            filters = {'member_of': parsed_args.uuid}
            url = common.url_with_filters(RP_BASE_URL, filters)
            rps = http.request('GET', url).json()['resource_providers']
            if not rps:
                raise exceptions.CommandError('No resource providers found in aggregate with uuid %s.' % parsed_args.uuid)
        else:
            url = RP_BASE_URL + '/' + parsed_args.uuid
            rps = [http.request('GET', url).json()]
        resources_list = []
        ret = 0
        for rp in rps:
            inventories = collections.defaultdict(dict)
            url = BASE_URL.format(uuid=rp['uuid'])
            if parsed_args.amend:
                payload = http.request('GET', url).json()
                inventories.update(payload['inventories'])
                payload['inventories'] = inventories
            else:
                payload = {'inventories': inventories, 'resource_provider_generation': rp['generation']}
            for r in parsed_args.resource:
                name, field, value = parse_resource_argument(r)
                inventories[name][field] = value
            try:
                if not parsed_args.dry_run:
                    resources = http.request('PUT', url, json=payload).json()
                else:
                    resources = payload
            except Exception as exp:
                with excutils.save_and_reraise_exception() as err_ctx:
                    if parsed_args.aggregate:
                        self.log.error(_('Failed to set inventory for resource provider %(rp)s: %(exp)s.'), {'rp': rp['uuid'], 'exp': exp})
                        err_ctx.reraise = False
                        ret += 1
                        continue
            resources_list.append((rp['uuid'], resources))
        if ret > 0:
            msg = _('Failed to set inventory for %(ret)s of %(total)s resource providers.') % {'ret': ret, 'total': len(rps)}
            raise exceptions.CommandError(msg)

        def get_rows(fields, resources, rp_uuid=None):
            inventories = [dict(resource_class=k, **v) for k, v in resources['inventories'].items()]
            prepend = (rp_uuid,) if rp_uuid else ()
            rows = (prepend + utils.get_dict_properties(i, fields) for i in inventories)
            return rows
        fields = ('resource_class',) + FIELDS
        if parsed_args.aggregate:
            rows = ()
            for rp_uuid, resources in resources_list:
                subrows = get_rows(fields, resources, rp_uuid=rp_uuid)
                rows = itertools.chain(rows, subrows)
            fields = ('resource_provider',) + fields
            return (fields, rows)
        else:
            return (fields, get_rows(fields, resources_list[0][1]))