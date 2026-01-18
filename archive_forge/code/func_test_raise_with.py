import string
from taskflow import exceptions as exc
from taskflow import test
def test_raise_with(self):
    capture = None
    try:
        raise IOError('broken')
    except Exception:
        try:
            exc.raise_with_cause(exc.TaskFlowException, 'broken')
        except Exception as e:
            capture = e
    self.assertIsNotNone(capture)
    self.assertIsInstance(capture, exc.TaskFlowException)
    self.assertIsNotNone(capture.cause)
    self.assertIsInstance(capture.cause, IOError)