import json
from oslo_metrics import message_type
from oslotest import base
def assertRaisesWithMessage(self, message, func, *args, **kwargs):
    try:
        func(*args, **kwargs)
        self.assertFail()
    except Exception as e:
        self.assertEqual(message, e.message)