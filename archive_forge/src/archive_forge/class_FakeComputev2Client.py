import copy
import random
from unittest import mock
import uuid
from novaclient import api_versions
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate as _aggregate
from openstack.compute.v2 import availability_zone as _availability_zone
from openstack.compute.v2 import extension as _extension
from openstack.compute.v2 import flavor as _flavor
from openstack.compute.v2 import hypervisor as _hypervisor
from openstack.compute.v2 import keypair as _keypair
from openstack.compute.v2 import migration as _migration
from openstack.compute.v2 import server as _server
from openstack.compute.v2 import server_action as _server_action
from openstack.compute.v2 import server_group as _server_group
from openstack.compute.v2 import server_interface as _server_interface
from openstack.compute.v2 import server_migration as _server_migration
from openstack.compute.v2 import service as _service
from openstack.compute.v2 import usage as _usage
from openstack.compute.v2 import volume_attachment as _volume_attachment
from openstackclient.api import compute_v2
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class FakeComputev2Client(object):

    def __init__(self, **kwargs):
        self.agents = mock.Mock()
        self.agents.resource_class = fakes.FakeResource(None, {})
        self.images = mock.Mock()
        self.images.resource_class = fakes.FakeResource(None, {})
        self.limits = mock.Mock()
        self.limits.resource_class = fakes.FakeResource(None, {})
        self.servers = mock.Mock()
        self.servers.resource_class = fakes.FakeResource(None, {})
        self.services = mock.Mock()
        self.services.resource_class = fakes.FakeResource(None, {})
        self.extensions = mock.Mock()
        self.extensions.resource_class = fakes.FakeResource(None, {})
        self.flavors = mock.Mock()
        self.flavor_access = mock.Mock()
        self.flavor_access.resource_class = fakes.FakeResource(None, {})
        self.quotas = mock.Mock()
        self.quotas.resource_class = fakes.FakeResource(None, {})
        self.quota_classes = mock.Mock()
        self.quota_classes.resource_class = fakes.FakeResource(None, {})
        self.usage = mock.Mock()
        self.usage.resource_class = fakes.FakeResource(None, {})
        self.volumes = mock.Mock()
        self.volumes.resource_class = fakes.FakeResource(None, {})
        self.hypervisors = mock.Mock()
        self.hypervisors.resource_class = fakes.FakeResource(None, {})
        self.hypervisors_stats = mock.Mock()
        self.hypervisors_stats.resource_class = fakes.FakeResource(None, {})
        self.keypairs = mock.Mock()
        self.keypairs.resource_class = fakes.FakeResource(None, {})
        self.hosts = mock.Mock()
        self.hosts.resource_class = fakes.FakeResource(None, {})
        self.server_groups = mock.Mock()
        self.server_groups.resource_class = fakes.FakeResource(None, {})
        self.server_migrations = mock.Mock()
        self.server_migrations.resource_class = fakes.FakeResource(None, {})
        self.instance_action = mock.Mock()
        self.instance_action.resource_class = fakes.FakeResource(None, {})
        self.migrations = mock.Mock()
        self.migrations.resource_class = fakes.FakeResource(None, {})
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']
        self.api_version = api_versions.APIVersion('2.1')