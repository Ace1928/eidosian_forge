import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def test_render_truncated(self):
    out, view = self.make_view()
    task_a = ProgressTask(None, progress_view=view)
    task_a.update('start_' + 'a' * 200 + '_end', 2000, 5000)
    line = view._render_line()
    self.assertEqual('- start_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.. 2000/5000', line)
    self.assertEqual(len(line), 79)