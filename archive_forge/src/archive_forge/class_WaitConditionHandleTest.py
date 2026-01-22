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
class WaitConditionHandleTest(common.HeatTestCase):

    def create_stack(self, stack_name=None, stack_id=None):
        temp = template_format.parse(test_template_waitcondition)
        template = tmpl.Template(temp)
        ctx = utils.dummy_context(tenant_id='test_tenant')
        if stack_name is None:
            stack_name = utils.random_name()
        stack = parser.Stack(ctx, stack_name, template, disable_rollback=True)
        if stack_id is not None:
            with utils.UUIDStub(stack_id):
                stack.store()
        else:
            stack.store()
        self.stack_id = stack.id
        with mock.patch.object(aws_wch.WaitConditionHandle, 'get_status') as m_gs:
            m_gs.return_value = ['SUCCESS']
            res_id = identifier.ResourceIdentifier('test_tenant', stack.name, stack.id, '', 'WaitHandle')
            with mock.patch.object(aws_wch.WaitConditionHandle, 'identifier') as m_id:
                m_id.return_value = res_id
                stack.create()
        rsrc = stack['WaitHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(rsrc.resource_id, rsrc.data().get('user_id'))
        return stack

    def test_handle(self):
        stack_id = 'STACKABCD1234'
        stack_name = 'test_stack2'
        now = datetime.datetime(2012, 11, 29, 13, 49, 37)
        timeutils.set_time_override(now)
        self.addCleanup(timeutils.clear_time_override)
        self.stack = self.create_stack(stack_id=stack_id, stack_name=stack_name)
        m_get_cfn_url = mock.Mock(return_value='http://server.test:8000/v1')
        self.stack.clients.client_plugin('heat').get_heat_cfn_url = m_get_cfn_url
        rsrc = self.stack['WaitHandle']
        self.assertEqual(rsrc.resource_id, rsrc.data().get('user_id'))
        rsrc.data_set('ec2_signed_url', None, False)
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        connection_url = ''.join(['http://server.test:8000/v1/waitcondition/', 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant%3Astacks%2F', 'test_stack2%2F', stack_id, '%2Fresources%2F', 'WaitHandle?'])
        expected_url = ''.join([connection_url, 'Timestamp=2012-11-29T13%3A49%3A37Z&', 'SignatureMethod=HmacSHA256&', 'AWSAccessKeyId=4567&', 'SignatureVersion=2&', 'Signature=', 'fHyt3XFnHq8%2FSwYaVcHdJka1hz6jdK5mHtgbo8OOKbQ%3D'])
        actual_url = rsrc.FnGetRefId()
        expected_params = parse.parse_qs(expected_url.split('?', 1)[1])
        actual_params = parse.parse_qs(actual_url.split('?', 1)[1])
        self.assertEqual(expected_params, actual_params)
        self.assertTrue(connection_url.startswith(connection_url))
        self.assertEqual(1, m_get_cfn_url.call_count)

    def test_handle_signal(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
        rsrc.handle_signal(test_metadata)
        handle_metadata = {u'123': {u'Data': u'foo', u'Reason': u'bar', u'Status': u'SUCCESS'}}
        self.assertEqual(handle_metadata, rsrc.metadata_get())

    def test_handle_signal_invalid(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        err_metadata = {'Data': 'foo', 'Status': 'SUCCESS', 'UniqueId': '123'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'UniqueId': '1234'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'UniqueId': '1234'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'data': 'foo', 'reason': 'bar', 'status': 'SUCCESS', 'uniqueid': '1234'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'UCCESS', 'UniqueId': '123'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'wibble', 'UniqueId': '123'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'success', 'UniqueId': '123'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)
        err_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'FAIL', 'UniqueId': '123'}
        self.assertRaises(ValueError, rsrc.handle_signal, err_metadata)

    def test_get_status(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        self.assertEqual([], rsrc.get_status())
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
        ret = rsrc.handle_signal(test_metadata)
        self.assertEqual(['SUCCESS'], rsrc.get_status())
        self.assertEqual('status:SUCCESS reason:bar', ret)
        test_metadata = {'Data': 'foo', 'Reason': 'bar2', 'Status': 'SUCCESS', 'UniqueId': '456'}
        ret = rsrc.handle_signal(test_metadata)
        self.assertEqual(['SUCCESS', 'SUCCESS'], rsrc.get_status())
        self.assertEqual('status:SUCCESS reason:bar2', ret)

    def test_get_status_reason(self):
        self.stack = self.create_stack()
        rsrc = self.stack['WaitHandle']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '123'}
        ret = rsrc.handle_signal(test_metadata)
        self.assertEqual(['bar'], rsrc.get_status_reason('SUCCESS'))
        self.assertEqual('status:SUCCESS reason:bar', ret)
        test_metadata = {'Data': 'dog', 'Reason': 'cat', 'Status': 'SUCCESS', 'UniqueId': '456'}
        ret = rsrc.handle_signal(test_metadata)
        self.assertEqual(['bar', 'cat'], sorted(rsrc.get_status_reason('SUCCESS')))
        self.assertEqual('status:SUCCESS reason:cat', ret)
        test_metadata = {'Data': 'boo', 'Reason': 'hoo', 'Status': 'FAILURE', 'UniqueId': '789'}
        ret = rsrc.handle_signal(test_metadata)
        self.assertEqual(['hoo'], rsrc.get_status_reason('FAILURE'))
        self.assertEqual('status:FAILURE reason:hoo', ret)