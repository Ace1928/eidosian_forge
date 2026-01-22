import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
class WaitConditionTest(common.HeatTestCase):

    def create_stack(self, stack_id=None, template=test_template_waitcondition, params=None, stub=True, stub_status=True):
        params = params or {}
        temp = template_format.parse(template)
        template = tmpl.Template(temp, env=environment.Environment(params))
        ctx = utils.dummy_context(tenant_id='test_tenant')
        stack = parser.Stack(ctx, 'test_stack', template, disable_rollback=True)
        if stack_id is None:
            stack_id = str(uuid.uuid4())
        self.stack_id = stack_id
        with utils.UUIDStub(self.stack_id):
            stack.store()
        if stub:
            res_id = identifier.ResourceIdentifier('test_tenant', stack.name, stack.id, '', 'WaitHandle')
            self.m_id = self.patchobject(aws_wch.WaitConditionHandle, 'identifier', return_value=res_id)
        if stub_status:
            self.m_gs = self.patchobject(aws_wch.WaitConditionHandle, 'get_status')
        return stack

    def test_post_success_to_handle(self):
        self.stack = self.create_stack()
        self.m_gs.side_effect = [[], [], ['SUCCESS']]
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        r = resource_objects.Resource.get_by_name_and_stack(self.stack.context, 'WaitHandle', self.stack.id)
        self.assertEqual('WaitHandle', r.name)
        self.assertEqual(3, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_post_failure_to_handle(self):
        self.stack = self.create_stack()
        self.m_gs.side_effect = [[], [], ['FAILURE']]
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        reason = rsrc.status_reason
        self.assertTrue(reason.startswith('WaitConditionFailure:'))
        r = resource_objects.Resource.get_by_name_and_stack(self.stack.context, 'WaitHandle', self.stack.id)
        self.assertEqual('WaitHandle', r.name)
        self.assertEqual(3, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_post_success_to_handle_count(self):
        self.stack = self.create_stack(template=test_template_wc_count)
        self.m_gs.side_effect = [[], ['SUCCESS'], ['SUCCESS', 'SUCCESS'], ['SUCCESS', 'SUCCESS', 'SUCCESS']]
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        r = resource_objects.Resource.get_by_name_and_stack(self.stack.context, 'WaitHandle', self.stack.id)
        self.assertEqual('WaitHandle', r.name)
        self.assertEqual(4, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_post_failure_to_handle_count(self):
        self.stack = self.create_stack(template=test_template_wc_count)
        self.m_gs.side_effect = [[], ['SUCCESS'], ['SUCCESS', 'FAILURE']]
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        reason = rsrc.status_reason
        self.assertTrue(reason.startswith('WaitConditionFailure:'))
        r = resource_objects.Resource.get_by_name_and_stack(self.stack.context, 'WaitHandle', self.stack.id)
        self.assertEqual('WaitHandle', r.name)
        self.assertEqual(3, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_timeout(self):
        self.stack = self.create_stack()
        m_ts = self.patchobject(self.stack, 'timeout_secs', return_value=None)
        self.m_gs.return_value = []
        now = timeutils.utcnow()
        periods = [0, 0.001, 0.1, 4.1, 5.1]
        periods.extend(range(10, 100, 5))
        fake_clock = [now + datetime.timedelta(0, t) for t in periods]
        timeutils.set_time_override(fake_clock)
        self.addCleanup(timeutils.clear_time_override)
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
        reason = rsrc.status_reason
        self.assertTrue(reason.startswith('WaitConditionTimeout:'))
        self.assertEqual(1, m_ts.call_count)
        self.assertEqual(1, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_FnGetAtt(self):
        self.stack = self.create_stack()
        self.m_gs.return_value = ['SUCCESS']
        self.stack.create()
        rsrc = self.stack['WaitForTheHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        wc_att = rsrc.FnGetAtt('Data')
        self.assertEqual(str({}), wc_att)
        handle = self.stack['WaitHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), handle.state)
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
        ret = handle.handle_signal(test_metadata)
        wc_att = rsrc.FnGetAtt('Data')
        self.assertEqual('{"123": "foo"}', wc_att)
        self.assertEqual('status:SUCCESS reason:bar', ret)
        test_metadata = {'Data': 'dog', 'Reason': 'cat', 'Status': 'SUCCESS', 'UniqueId': '456'}
        ret = handle.handle_signal(test_metadata)
        wc_att = rsrc.FnGetAtt('Data')
        self.assertIsInstance(wc_att, str)
        self.assertEqual({'123': 'foo', '456': 'dog'}, json.loads(wc_att))
        self.assertEqual('status:SUCCESS reason:cat', ret)
        self.assertEqual(1, self.m_gs.call_count)
        self.assertEqual(1, self.m_id.call_count)

    def test_FnGetRefId_resource_name(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        self.assertEqual('WaitHandle', rsrc.FnGetRefId())

    @mock.patch.object(aws_wch.WaitConditionHandle, '_get_ec2_signed_url')
    def test_FnGetRefId_signed_url(self, mock_get_signed_url):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        rsrc.resource_id = '123'
        mock_get_signed_url.return_value = 'http://signed_url'
        self.assertEqual('http://signed_url', rsrc.FnGetRefId())

    def test_FnGetRefId_convergence_cache_data(self):
        t = template_format.parse(test_template_waitcondition)
        template = tmpl.Template(t)
        stack = parser.Stack(utils.dummy_context(), 'test', template, cache_data={'WaitHandle': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'http://convg_signed_url'})})
        rsrc = stack.defn['WaitHandle']
        self.assertEqual('http://convg_signed_url', rsrc.FnGetRefId())

    def test_validate_handle_url_bad_stackid(self):
        stack_id = 'STACK_HUBSID_1234'
        t = json.loads(test_template_waitcondition)
        badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant' + '%3Astacks%2Ftest_stack%2F' + 'bad1' + '%2Fresources%2FWaitHandle'
        t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
        self.stack = self.create_stack(template=json.dumps(t), stub=False, stack_id=stack_id)
        rsrc = self.stack['WaitForTheHandle']
        self.assertRaises(ValueError, rsrc.handle_create)

    def test_validate_handle_url_bad_stackname(self):
        stack_id = 'STACKABCD1234'
        t = json.loads(test_template_waitcondition)
        badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant' + '%3Astacks%2FBAD_stack%2F' + stack_id + '%2Fresources%2FWaitHandle'
        t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
        self.stack = self.create_stack(template=json.dumps(t), stub=False, stack_id=stack_id)
        rsrc = self.stack['WaitForTheHandle']
        self.assertRaises(ValueError, rsrc.handle_create)

    def test_validate_handle_url_bad_tenant(self):
        stack_id = 'STACKABCD1234'
        t = json.loads(test_template_waitcondition)
        badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3ABAD_tenant' + '%3Astacks%2Ftest_stack%2F' + stack_id + '%2Fresources%2FWaitHandle'
        t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
        self.stack = self.create_stack(stack_id=stack_id, template=json.dumps(t), stub=False)
        rsrc = self.stack['WaitForTheHandle']
        self.assertRaises(ValueError, rsrc.handle_create)

    def test_validate_handle_url_bad_resource(self):
        stack_id = 'STACK_HUBR_1234'
        t = json.loads(test_template_waitcondition)
        badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant' + '%3Astacks%2Ftest_stack%2F' + stack_id + '%2Fresources%2FBADHandle'
        t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
        self.stack = self.create_stack(stack_id=stack_id, template=json.dumps(t), stub=False)
        rsrc = self.stack['WaitForTheHandle']
        self.assertRaises(ValueError, rsrc.handle_create)

    def test_validate_handle_url_bad_resource_type(self):
        stack_id = 'STACKABCD1234'
        t = json.loads(test_template_waitcondition)
        badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant' + '%3Astacks%2Ftest_stack%2F' + stack_id + '%2Fresources%2FWaitForTheHandle'
        t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
        self.stack = self.create_stack(stack_id=stack_id, template=json.dumps(t), stub=False)
        rsrc = self.stack['WaitForTheHandle']
        self.assertRaises(ValueError, rsrc.handle_create)