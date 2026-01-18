import unittest
from kivy.tests.common import GraphicUnitTest, UnitTestTouch
from kivy.base import EventLoop
from kivy.modules import inspector
from kivy.factory import Factory
def test_widget_popup(self, *args):
    EventLoop.ensure_window()
    self._win = EventLoop.window
    self.clean_garbage()
    self.root = self.builder.Builder.load_string(KV, filename='InspectorTestCase.KV')
    self.render(self.root)
    self.assertLess(len(self._win.children), 2)
    popup = self.root.ids.popup
    inspector.start(self._win, self.root)
    self.advance_frames(1)
    ins = self.root.inspector
    ins.inspect_enabled = False
    ins.activated = True
    self.assertTrue(ins.at_bottom)
    touch = UnitTestTouch(*popup.center)
    touch.touch_down()
    touch.touch_up()
    self.advance_frames(1)
    ins.inspect_enabled = True
    self.advance_frames(1)
    touch.touch_down()
    touch.touch_up()
    self.advance_frames(1)
    ins.show_widget_info()
    self.advance_frames(2)
    self.assertIsInstance(ins.widget, Factory.Button)
    self.assertIsInstance(ins.widget.parent, Factory.FirstModal)
    temp_popup = Factory.FirstModal()
    temp_popup_exp = temp_popup.ids.firstmodal.text
    self.assertEqual(ins.widget.text, temp_popup_exp)
    for node in ins.treeview.iterate_all_nodes():
        lkey = getattr(node.ids, 'lkey', None)
        if not lkey:
            continue
        if lkey.text == 'text':
            ltext = node.ids.ltext
            self.assertEqual(ltext.text[1:-1], temp_popup_exp)
            break
    del temp_popup
    ins.inspect_enabled = False
    touch = UnitTestTouch(0, 0)
    touch.touch_down()
    touch.touch_up()
    self.advance_frames(10)
    ins.activated = False
    self.render(self.root)
    self.advance_frames(5)
    inspector.stop(self._win, self.root)
    self.assertLess(len(self._win.children), 2)
    self.render(self.root)