import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class ListNetwork(neutronV20.ListCommand):
    """List networks that belong to a given tenant."""
    subnet_id_filter_len = 40
    marker_len = 44
    resource = 'network'
    _formatters = {'subnets': _format_subnets}
    list_columns = ['id', 'name', 'subnets']
    pagination_support = True
    sorting_support = True
    filter_attrs = ['tenant_id', 'name', 'admin_state_up', {'name': 'status', 'help': _('Filter %s according to their operation status.(For example: ACTIVE, ERROR etc)'), 'boolean': False, 'argparse_kwargs': {'type': utils.convert_to_uppercase}}, {'name': 'shared', 'help': _('Filter and list the networks which are shared.'), 'boolean': True}, {'name': 'router:external', 'help': _('Filter and list the networks which are external.'), 'boolean': True}, {'name': 'tags', 'help': _('Filter and list %s which has all given tags. Multiple tags can be set like --tags <tag[,tag...]>'), 'boolean': False, 'argparse_kwargs': {'metavar': 'TAG'}}, {'name': 'tags_any', 'help': _('Filter and list %s which has any given tags. Multiple tags can be set like --tags-any <tag[,tag...]>'), 'boolean': False, 'argparse_kwargs': {'metavar': 'TAG'}}, {'name': 'not_tags', 'help': _('Filter and list %s which does not have all given tags. Multiple tags can be set like --not-tags <tag[,tag...]>'), 'boolean': False, 'argparse_kwargs': {'metavar': 'TAG'}}, {'name': 'not_tags_any', 'help': _('Filter and list %s which does not have any given tags. Multiple tags can be set like --not-tags-any <tag[,tag...]>'), 'boolean': False, 'argparse_kwargs': {'metavar': 'TAG'}}]

    def extend_list(self, data, parsed_args):
        """Add subnet information to a network list."""
        neutron_client = self.get_client()
        search_opts = {'fields': ['id', 'cidr']}
        if self.pagination_support:
            page_size = parsed_args.page_size
            if page_size:
                search_opts.update({'limit': page_size})
        subnet_ids = []
        for n in data:
            if 'subnets' in n:
                subnet_ids.extend(n['subnets'])

        def _get_subnet_list(sub_ids):
            search_opts['id'] = sub_ids
            return neutron_client.list_subnets(**search_opts).get('subnets', [])
        try:
            subnets = _get_subnet_list(subnet_ids)
        except exceptions.RequestURITooLong as uri_len_exc:
            subnet_count = len(subnet_ids)
            max_size = self.subnet_id_filter_len * subnet_count - uri_len_exc.excess
            if self.pagination_support:
                max_size -= self.marker_len
            chunk_size = max_size // self.subnet_id_filter_len
            subnets = []
            for i in range(0, subnet_count, chunk_size):
                subnets.extend(_get_subnet_list(subnet_ids[i:i + chunk_size]))
        subnet_dict = dict([(s['id'], s) for s in subnets])
        for n in data:
            if 'subnets' in n:
                n['subnets'] = [subnet_dict.get(s) or {'id': s} for s in n['subnets']]