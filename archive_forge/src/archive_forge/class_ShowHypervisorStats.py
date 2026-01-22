from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowHypervisorStats(command.ShowOne):
    _description = _('Display hypervisor stats details')

    def take_action(self, parsed_args):
        self.log.warning(_('This command is deprecated.'))
        compute_client = self.app.client_manager.sdk_connection.compute
        response = compute_client.get('/os-hypervisors/statistics', microversion='2.1')
        hypervisor_stats = response.json().get('hypervisor_statistics')
        display_columns, columns = _get_hypervisor_stat_columns(hypervisor_stats)
        data = utils.get_dict_properties(hypervisor_stats, columns)
        return (display_columns, data)