import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
class CompositeAlarmTest(common.HeatTestCase):

    def setUp(self):
        super(CompositeAlarmTest, self).setUp()
        self.fa = mock.Mock()

    def create_stack(self, template=None):
        temp = template_format.parse(template)
        template = tmpl.Template(temp)
        ctx = utils.dummy_context()
        ctx.tenant = 'test_tenant'
        stack = parser.Stack(ctx, utils.random_name(), template, disable_rollback=True)
        stack.store()
        self.patchobject(aodh.AodhClientPlugin, '_create').return_value = self.fa
        self.patchobject(self.fa.alarm, 'create').return_value = FakeCompositeAlarm
        return stack

    def test_handle_create(self):
        """Test create the composite alarm."""
        test_stack = self.create_stack(template=alarm_template)
        test_stack.create()
        rsrc = test_stack['cps_alarm']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)

    def test_handle_update(self):
        """Test update the composite alarm."""
        test_stack = self.create_stack(template=alarm_template)
        update_mock = self.patchobject(self.fa.alarm, 'update')
        test_stack.create()
        rsrc = test_stack['cps_alarm']
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        after_props = copy.deepcopy(rsrc.properties.data)
        update_props = {'enabled': False, 'repeat_actions': False, 'insufficient_data_actions': [], 'ok_actions': ['signal_handler']}
        after_props.update(update_props)
        snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), after_props)
        scheduler.TaskRunner(rsrc.update, snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(1, update_mock.call_count)

    def test_validate(self):
        test_stack = self.create_stack(template=alarm_template)
        props = test_stack.t['resources']['cps_alarm']['Properties']
        props['composite_rule']['operator'] = 'invalid'
        res = test_stack['cps_alarm']
        error_msg = '"invalid" is not an allowed value'
        exc = self.assertRaises(exception.StackValidationFailed, res.validate)
        self.assertIn(error_msg, str(exc))

    def test_show_resource(self):
        test_stack = self.create_stack(template=alarm_template)
        res = test_stack['cps_alarm']
        res.client().alarm.create.return_value = FakeCompositeAlarm
        res.client().alarm.get.return_value = FakeCompositeAlarm
        scheduler.TaskRunner(res.create)()
        self.assertEqual(FakeCompositeAlarm, res.FnGetAtt('show'))