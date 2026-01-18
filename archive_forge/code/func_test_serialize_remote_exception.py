import sys
from oslo_serialization import jsonutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import common as exceptions
from oslo_messaging.tests import utils as test_utils
def test_serialize_remote_exception(self):
    try:
        try:
            raise self.cls(*self.args, **self.kwargs)
        except Exception as ex:
            cls_error = ex
            if self.add_remote:
                ex = add_remote_postfix(ex)
            raise ex
    except Exception:
        exc_info = sys.exc_info()
    serialized = exceptions.serialize_remote_exception(exc_info)
    failure = jsonutils.loads(serialized)
    self.assertEqual(self.clsname, failure['class'], failure)
    self.assertEqual(self.modname, failure['module'])
    self.assertEqual(self.msg, failure['message'])
    self.assertEqual([self.msg], failure['args'])
    self.assertEqual(self.kwargs, failure['kwargs'])
    tb = cls_error.__class__.__name__ + ': ' + self.msg
    self.assertIn(tb, ''.join(failure['tb']))