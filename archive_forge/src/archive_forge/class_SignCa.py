import os
from magnumclient.i18n import _
from osc_lib.command import command
class SignCa(command.Command):
    _description = _('Generate the CA certificate for a cluster.')

    def get_parser(self, prog_name):
        parser = super(SignCa, self).get_parser(prog_name)
        parser.add_argument('cluster', metavar='<cluster>', help='ID or name of the cluster')
        parser.add_argument('csr', metavar='<csr>', help='File path of csr file to send to Magnum to get signed.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        mag_client = self.app.client_manager.container_infra
        opts = {'cluster_uuid': _get_target_uuid(mag_client, parsed_args)}
        if parsed_args.csr is None or not os.path.isfile(parsed_args.csr):
            print('A CSR must be provided.')
            return
        with open(parsed_args.csr, 'r') as f:
            opts['csr'] = f.read()
        cert = mag_client.certificates.create(**opts)
        _show_cert(cert)