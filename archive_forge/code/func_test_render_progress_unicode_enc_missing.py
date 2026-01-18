import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def test_render_progress_unicode_enc_missing(self):
    out = codecs.getwriter('ascii')(io.BytesIO())
    self.assertRaises(AttributeError, getattr, out, 'encoding')
    view = self.make_view_only(out, 20)
    task = self.make_task(None, view, '§', 0, 1)
    view.show_progress(task)
    self.assertEqual(b'\r/ ? 0/1             \r', out.getvalue())