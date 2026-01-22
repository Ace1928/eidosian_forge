import os
import os_client_config
from tempest.lib.cli import base
def _get_admin_clients(self):
    creds = credentials()
    clients = base.CLIClient(username=creds['username'], password=creds['password'], tenant_name=creds['project_name'], project_name=creds['project_name'], user_domain_id=creds['user_domain_id'], project_domain_id=creds['project_domain_id'], uri=creds['auth_url'], cli_dir=CLI_DIR)
    return clients