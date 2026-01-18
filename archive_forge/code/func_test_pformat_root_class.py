import string
from taskflow import exceptions as exc
from taskflow import test
def test_pformat_root_class(self):
    ex = exc.TaskFlowException('Broken')
    self.assertIn('TaskFlowException', ex.pformat(show_root_class=True))
    self.assertNotIn('TaskFlowException', ex.pformat(show_root_class=False))
    self.assertIn('Broken', ex.pformat(show_root_class=True))