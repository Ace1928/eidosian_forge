from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def _get_adopt_data_and_template(self, environment=None):
    template = {'heat_template_version': '2013-05-23', 'parameters': {'app_dbx': {'type': 'string'}}, 'resources': {'res1': {'type': 'GenericResourceType'}}}
    adopt_data = {'status': 'COMPLETE', 'name': 'rtrove1', 'environment': environment, 'template': template, 'action': 'CREATE', 'id': '8532f0d3-ea84-444e-b2bb-2543bb1496a4', 'resources': {'res1': {'status': 'COMPLETE', 'name': 'database_password', 'resource_id': 'yBpuUROjfGQ2gKOD', 'action': 'CREATE', 'type': 'GenericResourceType', 'metadata': {}}}}
    return (template, adopt_data)