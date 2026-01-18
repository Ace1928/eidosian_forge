import unittest
from kivy.tests.common import GraphicUnitTest, UnitTestTouch
from kivy.base import EventLoop
from kivy.modules import inspector
from kivy.factory import Factory
def test_widget_button(self, *args):
    EventLoop.ensure_window()
    self._win = EventLoop.window
    self.clean_garbage()
    self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
    self.render(self.root)
    self.assertLess(len(self._win.children), 2)
    highlight = self.root.ids.highlight
    highlight_exp = self.root.ids.highlight.text
    inspector.start(self._win, self.root)
    self.advance_frames(2)
    ins = self.root.inspector
    ins.activated = True
    ins.inspect_enabled = True
    self.assertTrue(ins.at_bottom)
    touch = UnitTestTouch(*highlight.center)
    touch.touch_down()
    touch.touch_up()
    ins.show_widget_info()
    self.advance_frames(2)
    self.assertEqual(ins.widget.text, highlight_exp)
    for node in ins.treeview.iterate_all_nodes():
        lkey = getattr(node.ids, 'lkey', None)
        if not lkey:
            continue
        if lkey.text == 'text':
            ltext = node.ids.ltext
            self.assertEqual(ltext.text[1:-1], highlight_exp)
            break
    ins.inspect_enabled = False
    ins.activated = False
    self.render(self.root)
    self.advance_frames(1)
    inspector.stop(self._win, self.root)
    self.assertLess(len(self._win.children), 2)
    self.render(self.root)