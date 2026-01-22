import os
import time
from cinderclient.v3 import client as cinderclient
import fixtures
from glanceclient import client as glanceclient
from keystoneauth1.exceptions import discovery as discovery_exc
from keystoneauth1 import identity
from keystoneauth1 import session as ksession
from keystoneclient import client as keystoneclient
from keystoneclient import discover as keystone_discover
from neutronclient.v2_0 import client as neutronclient
import openstack.config
import openstack.config.exceptions
from oslo_utils import uuidutils
import tempest.lib.cli.base
import testtools
import novaclient
import novaclient.api_versions
from novaclient import base
import novaclient.client
from novaclient.v2 import networks
import novaclient.v2.shell
class TenantTestBase(ClientTestBase):
    """Base test class for additional tenant and user creation which
    could be required in various test scenarios
    """

    def setUp(self):
        super(TenantTestBase, self).setUp()
        user_name = uuidutils.generate_uuid()
        project_name = uuidutils.generate_uuid()
        password = 'password'
        if self.keystone.version == 'v3':
            project = self.keystone.projects.create(project_name, self.project_domain_id)
            self.project_id = project.id
            self.addCleanup(self.keystone.projects.delete, self.project_id)
            self.user_id = self.keystone.users.create(name=user_name, password=password, default_project=self.project_id).id
            for role in self.keystone.roles.list():
                if 'member' in role.name.lower():
                    self.keystone.roles.grant(role.id, user=self.user_id, project=self.project_id)
                    break
        else:
            project = self.keystone.tenants.create(project_name)
            self.project_id = project.id
            self.addCleanup(self.keystone.tenants.delete, self.project_id)
            self.user_id = self.keystone.users.create(user_name, password, tenant_id=self.project_id).id
        self.addCleanup(self.keystone.users.delete, self.user_id)
        self.cli_clients_2 = tempest.lib.cli.base.CLIClient(username=user_name, password=password, tenant_name=project_name, uri=self.cli_clients.uri, cli_dir=self.cli_clients.cli_dir, insecure=self.insecure)

    def another_nova(self, action, flags='', params='', fail_ok=False, endpoint_type='publicURL', merge_stderr=False):
        flags += ' --os-compute-api-version %s ' % self.COMPUTE_API_VERSION
        return self.cli_clients_2.nova(action, flags, params, fail_ok, endpoint_type, merge_stderr)