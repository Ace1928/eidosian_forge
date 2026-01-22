import argparse
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import exceptions as nc_exc
class ListSfcFlowClassifier(command.Lister):
    _description = _('List flow classifiers')

    def get_parser(self, prog_name):
        parser = super(ListSfcFlowClassifier, self).get_parser(prog_name)
        parser.add_argument('--long', action='store_true', help=_('List additional fields in output'))
        return parser

    def extend_list(self, data, parsed_args):
        ext_data = []
        for d in data:
            val = []
            protocol = d['protocol'].upper() if d['protocol'] else 'any'
            val.append('protocol: ' + protocol)
            val.append(self._get_protocol_port_details(d, 'source'))
            val.append(self._get_protocol_port_details(d, 'destination'))
            if 'logical_source_port' in d:
                val.append('neutron_source_port: ' + str(d['logical_source_port']))
            if 'logical_destination_port' in d:
                val.append('neutron_destination_port: ' + str(d['logical_destination_port']))
            if 'l7_parameters' in d:
                l7_param = 'l7_parameters: {%s}' % ','.join(d['l7_parameters'])
                val.append(l7_param)
            d['summary'] = ',\n'.join(val)
            ext_data.append(d)
        return ext_data

    def _get_protocol_port_details(self, data, val):
        type_ip_prefix = val + '_ip_prefix'
        ip_prefix = data.get(type_ip_prefix)
        if not ip_prefix:
            ip_prefix = 'any'
        min_port = data.get(val + '_port_range_min')
        if min_port is None:
            min_port = 'any'
        max_port = data.get(val + '_port_range_max')
        if max_port is None:
            max_port = 'any'
        return '%s[port]: %s[%s:%s]' % (val, ip_prefix, min_port, max_port)

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.sfc_flow_classifiers()
        obj_extend = self.extend_list(obj, parsed_args)
        headers, columns = column_util.get_column_definitions(_attr_map, long_listing=parsed_args.long)
        return (headers, (utils.get_dict_properties(s, columns) for s in obj_extend))