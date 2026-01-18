import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def test_render_progress_sub_nested(self):
    """Intermediate tasks don't mess up calculation."""
    out, view = self.make_view()
    view.enable_bar = True
    task_a = ProgressTask(None, progress_view=view)
    task_a.update('a', 0, 2)
    task_b = ProgressTask(task_a, progress_view=view)
    task_b.update('b')
    task_c = ProgressTask(task_b, progress_view=view)
    task_c.update('c', 1, 2)
    self.assertEqual('[####|               ] a:b:c 1/2                                               ', view._render_line())