import os
import os_client_config
from tempest.lib.cli import base
class MistralCLIAuth(base.ClientTestBase):
    _mistral_url = None

    def _get_admin_clients(self):
        creds = credentials()
        clients = base.CLIClient(username=creds['username'], password=creds['password'], tenant_name=creds['project_name'], project_name=creds['project_name'], user_domain_id=creds['user_domain_id'], project_domain_id=creds['project_domain_id'], uri=creds['auth_url'], cli_dir=CLI_DIR)
        return clients

    def _get_clients(self):
        return self._get_admin_clients()

    def mistral(self, action, flags='', params='', fail_ok=False):
        """Executes Mistral command."""
        mistral_url_op = '--os-mistral-url %s' % self._mistral_url
        flags = '{} --insecure'.format(flags)
        if 'WITHOUT_AUTH' in os.environ:
            return base.execute('mistral %s' % mistral_url_op, action, flags, params, fail_ok, merge_stderr=False, cli_dir='')
        else:
            return self.clients.cmd_with_auth('mistral %s' % mistral_url_op, action, flags, params, fail_ok)

    def get_project_id(self, project_name='admin'):
        admin_clients = self._get_clients()
        projects = self.parser.listing(admin_clients.openstack('project show', params=project_name, flags='--os-identity-api-version 3 --insecure'))
        return [o['Value'] for o in projects if o['Field'] == 'id'][0]