import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SoftwareDeploymentGroupAttrTest(common.HeatTestCase):
    scenarios = [('stdouts', dict(group_attr='deploy_stdouts', nested_attr='deploy_stdout', values=['Thing happened on server1', 'ouch'])), ('stderrs', dict(group_attr='deploy_stderrs', nested_attr='deploy_stderr', values=['', "It's gone Pete Tong"])), ('status_codes', dict(group_attr='deploy_status_codes', nested_attr='deploy_status_code', values=[0, 1])), ('passthrough', dict(group_attr='some_attr', nested_attr='some_attr', values=['attr1', 'attr2']))]
    template = {'heat_template_version': '2013-05-23', 'resources': {'deploy_mysql': {'type': 'OS::Heat::SoftwareDeploymentGroup', 'properties': {'config': 'config_uuid', 'servers': {'server1': 'uuid1', 'server2': 'uuid2'}, 'input_values': {'foo': 'bar'}, 'name': '10_config'}}}}

    def setUp(self):
        super(SoftwareDeploymentGroupAttrTest, self).setUp()
        self.server_names = ['server1', 'server2']
        self.servers = [mock.MagicMock() for s in self.server_names]
        self.stack = utils.parse_stack(self.template)

    def test_attributes(self):
        resg = self.create_dummy_stack()
        self.assertEqual(dict(zip(self.server_names, self.values)), resg.FnGetAtt(self.group_attr))
        self.check_calls()

    def test_attributes_path(self):
        resg = self.create_dummy_stack()
        for i, r in enumerate(self.server_names):
            self.assertEqual(self.values[i], resg.FnGetAtt(self.group_attr, r))
        self.check_calls(len(self.server_names))

    def create_dummy_stack(self):
        snip = self.stack.t.resource_definitions(self.stack)['deploy_mysql']
        resg = sd.SoftwareDeploymentGroup('test', snip, self.stack)
        resg.resource_id = 'test-test'
        nested = self.patchobject(resg, 'nested')
        nested.return_value = dict(zip(self.server_names, self.servers))
        self._stub_get_attr(resg)
        return resg

    def _stub_get_attr(self, resg):

        def ref_id_fn(args):
            self.fail('Getting member reference ID for some reason')

        def attr_fn(args):
            res_name = args[0]
            return self.values[self.server_names.index(res_name)]

        def get_output(output_name):
            outputs = resg._nested_output_defns(resg._resource_names(), attr_fn, ref_id_fn)
            op_defns = {od.name: od for od in outputs}
            self.assertIn(output_name, op_defns)
            return op_defns[output_name].get_value()
        orig_get_attr = resg.FnGetAtt

        def get_attr(attr_name, *path):
            if not path:
                attr = attr_name
            else:
                attr = (attr_name,) + path
            resg.referenced_attrs = mock.Mock(return_value=[attr])
            return orig_get_attr(attr_name, *path)
        resg.FnGetAtt = mock.Mock(side_effect=get_attr)
        resg.get_output = mock.Mock(side_effect=get_output)

    def check_calls(self, count=1):
        pass