import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class ListSecurityGroupRule(neutronV20.ListCommand):
    """List security group rules that belong to a given tenant."""
    resource = 'security_group_rule'
    list_columns = ['id', 'security_group_id', 'direction', 'ethertype', 'port/protocol', 'remote']
    replace_rules = {'security_group_id': 'security_group', 'remote_group_id': 'remote_group'}
    digest_fields = {'remote': {'method': _get_remote, 'depends_on': ['remote_ip_prefix', 'remote_group_id']}, 'port/protocol': {'method': _get_protocol_port, 'depends_on': ['protocol', 'port_range_min', 'port_range_max']}, 'protocol/port': {'method': _get_protocol_port, 'depends_on': ['protocol', 'port_range_min', 'port_range_max']}}
    pagination_support = True
    sorting_support = True

    def get_parser(self, prog_name):
        parser = super(ListSecurityGroupRule, self).get_parser(prog_name)
        parser.add_argument('--no-nameconv', action='store_true', help=_('Do not convert security group ID to its name.'))
        return parser

    @staticmethod
    def replace_columns(cols, rules, reverse=False):
        if reverse:
            rules = dict(((rules[k], k) for k in rules.keys()))
        return [rules.get(col, col) for col in cols]

    def get_required_fields(self, fields):
        fields = self.replace_columns(fields, self.replace_rules, reverse=True)
        for field, digest_fields in self.digest_fields.items():
            if field in fields:
                fields += digest_fields['depends_on']
                fields.remove(field)
        return fields

    def retrieve_list(self, parsed_args):
        parsed_args.fields = self.get_required_fields(parsed_args.fields)
        return super(ListSecurityGroupRule, self).retrieve_list(parsed_args)

    def _get_sg_name_dict(self, data, page_size, no_nameconv):
        """Get names of security groups referred in the retrieved rules.

        :return: a dict from secgroup ID to secgroup name
        """
        if no_nameconv:
            return {}
        neutron_client = self.get_client()
        search_opts = {'fields': ['id', 'name']}
        if self.pagination_support:
            if page_size:
                search_opts.update({'limit': page_size})
        sec_group_ids = set()
        for rule in data:
            for key in self.replace_rules:
                if rule.get(key):
                    sec_group_ids.add(rule[key])
        sec_group_ids = list(sec_group_ids)

        def _get_sec_group_list(sec_group_ids):
            search_opts['id'] = sec_group_ids
            return neutron_client.list_security_groups(**search_opts).get('security_groups', [])
        try:
            secgroups = _get_sec_group_list(sec_group_ids)
        except exceptions.RequestURITooLong as uri_len_exc:
            sec_group_id_filter_len = 40
            sec_group_count = len(sec_group_ids)
            max_size = sec_group_id_filter_len * sec_group_count - uri_len_exc.excess
            chunk_size = max_size // sec_group_id_filter_len
            secgroups = []
            for i in range(0, sec_group_count, chunk_size):
                secgroups.extend(_get_sec_group_list(sec_group_ids[i:i + chunk_size]))
        return dict([(sg['id'], sg['name']) for sg in secgroups if sg['name']])

    @staticmethod
    def _has_fields(rule, required_fields):
        return all([key in rule for key in required_fields])

    def extend_list(self, data, parsed_args):
        sg_dict = self._get_sg_name_dict(data, parsed_args.page_size, parsed_args.no_nameconv)
        for rule in data:
            for key in self.replace_rules:
                if key in rule:
                    rule[key] = sg_dict.get(rule[key], rule[key])
            for field, digest_rule in self.digest_fields.items():
                if self._has_fields(rule, digest_rule['depends_on']):
                    rule[field] = digest_rule['method'](rule) or 'any'

    def setup_columns(self, info, parsed_args):
        parsed_args.columns = self.replace_columns(parsed_args.columns, self.replace_rules, reverse=True)
        info = super(ListSecurityGroupRule, self).setup_columns(info, parsed_args)
        cols = info[0]
        if not parsed_args.no_nameconv:
            cols = self.replace_columns(info[0], self.replace_rules)
            parsed_args.columns = cols
        return (cols, info[1])