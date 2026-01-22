import copy
import datetime
import random
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
class FakeShareClient(object):

    def __init__(self, **kwargs):
        super(FakeShareClient, self).__init__()
        self.auth_token = kwargs['token']
        self.management_url = kwargs['endpoint']
        self.shares = mock.Mock()
        self.transfers = mock.Mock()
        self.share_access_rules = mock.Mock()
        self.share_groups = mock.Mock()
        self.share_types = mock.Mock()
        self.share_type_access = mock.Mock()
        self.quotas = mock.Mock()
        self.quota_classes = mock.Mock()
        self.share_backups = mock.Mock()
        self.share_snapshots = mock.Mock()
        self.share_group_snapshots = mock.Mock()
        self.share_snapshot_export_locations = mock.Mock()
        self.share_snapshot_instances = mock.Mock()
        self.share_replicas = mock.Mock()
        self.share_replica_export_locations = mock.Mock()
        self.share_networks = mock.Mock()
        self.share_network_subnets = mock.Mock()
        self.security_services = mock.Mock()
        self.shares.resource_class = osc_fakes.FakeResource(None, {})
        self.share_instance_export_locations = mock.Mock()
        self.share_export_locations = mock.Mock()
        self.share_snapshot_instance_export_locations = mock.Mock()
        self.share_export_locations.resource_class = osc_fakes.FakeResource(None, {})
        self.messages = mock.Mock()
        self.availability_zones = mock.Mock()
        self.services = mock.Mock()
        self.share_instances = mock.Mock()
        self.pools = mock.Mock()
        self.limits = mock.Mock()
        self.share_group_types = mock.Mock()
        self.share_group_type_access = mock.Mock()
        self.share_servers = mock.Mock()
        self.resource_locks = mock.Mock()