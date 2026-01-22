import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
class TestTextProgressView(tests.TestCase):
    """Tests for text display of progress bars.

    These try to exercise the progressview independently of its construction,
    which is arranged by the TextUIFactory.
    """

    def make_view_only(self, out, width=79):
        view = TextProgressView(out)
        view._avail_width = lambda: width
        return view

    def make_view(self):
        out = ui_testing.StringIOWithEncoding()
        return (out, self.make_view_only(out))

    def make_task(self, parent_task, view, msg, curr, total):
        task = ProgressTask(parent_task, progress_view=view)
        task.msg = msg
        task.current_cnt = curr
        task.total_cnt = total
        return task

    def test_clear(self):
        out, view = self.make_view()
        task = self.make_task(None, view, 'reticulating splines', 5, 20)
        view.show_progress(task)
        self.assertEqual('\r/ reticulating splines 5/20                                                    \r', out.getvalue())
        view.clear()
        self.assertEqual('\r/ reticulating splines 5/20                                                    \r' + '\r' + 79 * ' ' + '\r', out.getvalue())

    def test_render_progress_no_bar(self):
        """The default view now has a spinner but no bar."""
        out, view = self.make_view()
        task = self.make_task(None, view, 'reticulating splines', 5, 20)
        view.show_progress(task)
        self.assertEqual('\r/ reticulating splines 5/20                                                    \r', out.getvalue())

    def test_render_progress_easy(self):
        """Just one task and one quarter done"""
        out, view = self.make_view()
        view.enable_bar = True
        task = self.make_task(None, view, 'reticulating splines', 5, 20)
        view.show_progress(task)
        self.assertEqual('\r[####/               ] reticulating splines 5/20                               \r', out.getvalue())

    def test_render_progress_nested(self):
        """Tasks proportionally contribute to overall progress"""
        out, view = self.make_view()
        task = self.make_task(None, view, 'reticulating splines', 0, 2)
        task2 = self.make_task(task, view, 'stage2', 1, 2)
        view.show_progress(task2)
        view.enable_bar = True
        self.assertEqual('[####-               ] reticulating splines:stage2 1/2                         ', view._render_line())
        task2.update('stage2', 2, 2)
        self.assertEqual('[#########\\          ] reticulating splines:stage2 2/2                         ', view._render_line())

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

    def test_render_truncated(self):
        out, view = self.make_view()
        task_a = ProgressTask(None, progress_view=view)
        task_a.update('start_' + 'a' * 200 + '_end', 2000, 5000)
        line = view._render_line()
        self.assertEqual('- start_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.. 2000/5000', line)
        self.assertEqual(len(line), 79)

    def test_render_with_activity(self):
        out, view = self.make_view()
        task_a = ProgressTask(None, progress_view=view)
        view._last_transport_msg = '   123kB   100kB/s '
        line = view._render_line()
        self.assertEqual('   123kB   100kB/s /                                                           ', line)
        self.assertEqual(len(line), 79)
        task_a.update('start_' + 'a' * 200 + '_end', 2000, 5000)
        view._last_transport_msg = '   123kB   100kB/s '
        line = view._render_line()
        self.assertEqual('   123kB   100kB/s \\ start_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.. 2000/5000', line)
        self.assertEqual(len(line), 79)

    def test_render_progress_unicode_enc_utf8(self):
        out = ui_testing.StringIOWithEncoding()
        out.encoding = 'utf-8'
        view = self.make_view_only(out, 20)
        task = self.make_task(None, view, '§', 0, 1)
        view.show_progress(task)
        self.assertEqual('\r/ § 0/1            \r', out.getvalue())

    def test_render_progress_unicode_enc_missing(self):
        out = codecs.getwriter('ascii')(io.BytesIO())
        self.assertRaises(AttributeError, getattr, out, 'encoding')
        view = self.make_view_only(out, 20)
        task = self.make_task(None, view, '§', 0, 1)
        view.show_progress(task)
        self.assertEqual(b'\r/ ? 0/1             \r', out.getvalue())

    def test_render_progress_unicode_enc_none(self):
        out = ui_testing.StringIOWithEncoding()
        out.encoding = None
        view = self.make_view_only(out, 20)
        task = self.make_task(None, view, '§', 0, 1)
        view.show_progress(task)
        self.assertEqual('\r/ ? 0/1             \r', out.getvalue())