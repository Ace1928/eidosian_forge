import json
from oslo_metrics import message_type
from oslotest import base
class TestMetricValidation(base.BaseTestCase):

    def setUp(self):
        super(TestMetricValidation, self).setUp()

    def assertRaisesWithMessage(self, message, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
            self.assertFail()
        except Exception as e:
            self.assertEqual(message, e.message)

    def test_message_validation(self):
        metric = dict()
        message = 'module should be specified'
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['module'] = 'test'
        message = 'name should be specified'
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['name'] = 'test'
        message = 'action should be specified'
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['action'] = 'test'
        message = 'labels should be specified'
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['labels'] = 'test_label'
        message = "action need 'value' field"
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['action'] = {'value': '1'}
        message = "action need 'action' field"
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))
        metric['action']['action'] = 'test'
        message = "action should be choosen from ['inc', 'observe']"
        self.assertRaisesWithMessage(message, message_type.Metric.from_json, json.dumps(metric))