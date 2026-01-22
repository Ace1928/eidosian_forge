import json
import re
from novaclient import exceptions as nova_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ShowHypervisor(command.ShowOne):
    _description = _('Display hypervisor details')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('hypervisor', metavar='<hypervisor>', help=_('Hypervisor to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        hypervisor = compute_client.find_hypervisor(parsed_args.hypervisor, ignore_missing=False).copy()
        aggregates = compute_client.aggregates()
        hypervisor['aggregates'] = list()
        service_details = hypervisor['service_details']
        if aggregates:
            if '@' in service_details['host']:
                cell, service_host = service_details['host'].split('@', 1)
            else:
                cell = None
                service_host = service_details['host']
            if cell:
                member_of = [aggregate.name for aggregate in aggregates if cell in aggregate.name and service_host in aggregate.hosts]
            else:
                member_of = [aggregate.name for aggregate in aggregates if service_host in aggregate.hosts]
            hypervisor['aggregates'] = member_of
        try:
            if sdk_utils.supports_microversion(compute_client, '2.88'):
                uptime = hypervisor['uptime'] or ''
                del hypervisor['uptime']
            else:
                del hypervisor['uptime']
                uptime = compute_client.get_hypervisor_uptime(hypervisor['id'])['uptime']
            m = re.match('\\s*(.+)\\sup\\s+(.+),\\s+(.+)\\susers?,\\s+load average:\\s(.+)', uptime)
            if m:
                hypervisor['host_time'] = m.group(1)
                hypervisor['uptime'] = m.group(2)
                hypervisor['users'] = m.group(3)
                hypervisor['load_average'] = m.group(4)
        except nova_exceptions.HTTPNotImplemented:
            pass
        hypervisor['service_id'] = service_details['id']
        hypervisor['service_host'] = service_details['host']
        del hypervisor['service_details']
        if not sdk_utils.supports_microversion(compute_client, '2.28'):
            hypervisor['cpu_info'] = json.loads(hypervisor['cpu_info'] or '{}')
        display_columns, columns = _get_hypervisor_columns(hypervisor, compute_client)
        data = utils.get_dict_properties(hypervisor, columns, formatters={'cpu_info': format_columns.DictColumn})
        return (display_columns, data)