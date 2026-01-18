import datetime
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import crypt
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import service
from heat.engine import service_software_config
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
@mock.patch.object(service_software_config.SoftwareConfigService, 'metadata_software_deployments')
@mock.patch.object(db_api, 'resource_update')
@mock.patch.object(db_api, 'resource_get_by_physical_resource_id')
@mock.patch.object(service_software_config.requests, 'put')
def test_push_metadata_software_deployments_retry(self, put, res_get, res_upd, md_sd):
    rs = mock.Mock()
    rs.rsrc_metadata = {'original': 'metadata'}
    rs.id = '1234'
    rs.atomic_key = 1
    rs.data = []
    res_get.return_value = rs
    res_upd.return_value = 0
    deployments = {'deploy': 'this'}
    md_sd.return_value = deployments
    f = self.engine.software_config._push_metadata_software_deployments
    self.patchobject(f.retry, 'sleep')
    self.assertRaises(exception.ConcurrentTransaction, f, self.ctx, '1234', None)
    self.assertEqual(11, res_upd.call_count)
    put.assert_not_called()