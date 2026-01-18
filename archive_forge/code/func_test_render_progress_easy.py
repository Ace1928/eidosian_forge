import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def test_render_progress_easy(self):
    """Just one task and one quarter done"""
    out, view = self.make_view()
    view.enable_bar = True
    task = self.make_task(None, view, 'reticulating splines', 5, 20)
    view.show_progress(task)
    self.assertEqual('\r[####/               ] reticulating splines 5/20                               \r', out.getvalue())