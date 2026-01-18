import string
from taskflow import exceptions as exc
from taskflow import test
def test_pformat_str(self):
    ex = None
    try:
        try:
            try:
                raise IOError("Didn't work")
            except IOError:
                exc.raise_with_cause(exc.TaskFlowException, "It didn't go so well")
        except exc.TaskFlowException:
            exc.raise_with_cause(exc.TaskFlowException, 'I Failed')
    except exc.TaskFlowException as e:
        ex = e
    self.assertIsNotNone(ex)
    self.assertIsInstance(ex, exc.TaskFlowException)
    self.assertIsInstance(ex.cause, exc.TaskFlowException)
    self.assertIsInstance(ex.cause.cause, IOError)
    p_msg = ex.pformat()
    p_str_msg = str(ex)
    for msg in ['I Failed', "It didn't go so well", "Didn't work"]:
        self.assertIn(msg, p_msg)
        self.assertIn(msg, p_str_msg)