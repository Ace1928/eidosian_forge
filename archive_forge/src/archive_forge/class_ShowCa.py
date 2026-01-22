import os
from magnumclient.i18n import _
from osc_lib.command import command
class ShowCa(command.Command):
    _description = _('Show details about the CA certificate for a cluster.')

    def get_parser(self, prog_name):
        parser = super(ShowCa, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help='ID or name of the cluster')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        cluster = mag_client.clusters.get(parsed_args.cluster)
        cert = mag_client.certificates.get(cluster.uuid)
        _show_cert(cert)