from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
class ShowClusterTemplate(command.ShowOne):
    """Show a Cluster Template."""
    _description = _('Show a Cluster Template.')
    log = logging.getLogger(__name__ + '.ShowClusterTemplate')

    def get_parser(self, prog_name):
        parser = super(ShowClusterTemplate, self).get_parser(prog_name)
        parser.add_argument('cluster-template', metavar='<cluster-template>', help=_('ID or name of the cluster template to show.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        ct = getattr(parsed_args, 'cluster-template', None)
        cluster_template = mag_client.cluster_templates.get(ct)
        return _show_cluster_template(cluster_template)