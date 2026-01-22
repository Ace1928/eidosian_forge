from kivy.tests.common import GraphicUnitTest, UTMotionEvent
from kivy.lang import Builder
from kivy.base import EventLoop
from kivy.weakproxy import WeakProxy
from time import sleep
class ActionBarTestCase(GraphicUnitTest):
    framecount = 0

    def setUp(self):
        global DropDown
        from kivy.uix.dropdown import DropDown
        import kivy.lang.builder as builder
        if not hasattr(self, '_trace'):
            self._trace = builder.trace
        self.builder = builder
        builder.trace = lambda *_, **__: None
        super(ActionBarTestCase, self).setUp()

    def tearDown(self, *args, **kwargs):
        import kivy.lang.builder as builder
        builder.trace = self._trace
        super(ActionBarTestCase, self).tearDown(*args, **kwargs)

    def move_frames(self, t):
        for i in range(t):
            EventLoop.idle()

    def clean_garbage(self, *args):
        for child in self._win.children[:]:
            self._win.remove_widget(child)
        self.move_frames(5)

    def check_dropdown(self, present=True):
        any_list = [isinstance(child, DropDown) for child in self._win.children]
        self.assertLess(sum(any_list), 2)
        if not present and (not any(any_list)):
            return
        elif present and any(any_list):
            return
        print("DropDown either missing, or isn't supposed to be there")
        self.assertTrue(False)

    def test_1_openclose(self, *args):
        self._win = EventLoop.window
        self.clean_garbage()
        root = Builder.load_string(KV)
        self.render(root)
        self.assertLess(len(self._win.children), 2)
        group2 = root.ids.group2
        group1 = root.ids.group1
        self.move_frames(5)
        self.check_dropdown(present=False)
        self.assertFalse(group2.is_open)
        self.assertFalse(group1.is_open)
        items = ((group2, group1), (group1, group2))
        for item in items:
            active, passive = item
            TouchPoint(*active.center)
            self.check_dropdown(present=True)
            gdd = WeakProxy(self._win.children[0])
            self.assertIn(gdd, self._win.children)
            self.assertEqual(gdd, self._win.children[0])
            self.assertTrue(active.is_open)
            self.assertFalse(passive.is_open)
            TouchPoint(0, 0)
            sleep(gdd.min_state_time)
            self.move_frames(1)
            self.assertNotEqual(gdd, self._win.children[0])
            self.assertLess(len(self._win.children), 2)
            self.check_dropdown(present=False)
            self.assertFalse(active.is_open)
            self.assertFalse(passive.is_open)
        self._win.remove_widget(root)

    def test_2_switch(self, *args):
        self._win = EventLoop.window
        self.clean_garbage()
        root = Builder.load_string(KV)
        self.render(root)
        self.assertLess(len(self._win.children), 2)
        group2 = root.ids.group2
        group1 = root.ids.group1
        self.move_frames(5)
        self.check_dropdown(present=False)
        self.assertFalse(group2.is_open)
        self.assertFalse(group1.is_open)
        TouchPoint(*group2.center)
        self.check_dropdown(present=True)
        g2dd = WeakProxy(self._win.children[0])
        self.assertIn(g2dd, self._win.children)
        self.assertEqual(g2dd, self._win.children[0])
        self.assertTrue(group2.is_open)
        self.assertFalse(group1.is_open)
        TouchPoint(0, 0)
        sleep(g2dd.min_state_time)
        self.move_frames(1)
        TouchPoint(*group1.center)
        sleep(g2dd.min_state_time)
        self.move_frames(1)
        self.assertNotEqual(g2dd, self._win.children[0])
        self.assertFalse(group2.is_open)
        self.assertTrue(group1.is_open)
        self.check_dropdown(present=True)
        TouchPoint(0, 0)
        sleep(g2dd.min_state_time)
        self.move_frames(1)
        self.check_dropdown(present=False)
        self.assertFalse(group2.is_open)
        self.assertFalse(group1.is_open)
        self.assertNotIn(g2dd, self._win.children)
        self._win.remove_widget(root)

    def test_3_openpress(self, *args):
        self._win = EventLoop.window
        self.clean_garbage()
        root = Builder.load_string(KV)
        self.render(root)
        self.assertLess(len(self._win.children), 2)
        group2 = root.ids.group2
        group2button = root.ids.group2button
        group1 = root.ids.group1
        group1button = root.ids.group1button
        self.move_frames(5)
        self.check_dropdown(present=False)
        self.assertFalse(group2.is_open)
        self.assertFalse(group1.is_open)
        items = ((group2, group1, group2button), (group1, group2, group1button))
        for item in items:
            active, passive, button = item
            TouchPoint(*active.center)
            self.check_dropdown(present=True)
            gdd = WeakProxy(self._win.children[0])
            self.assertIn(gdd, self._win.children)
            self.assertEqual(gdd, self._win.children[0])
            self.assertTrue(active.is_open)
            self.assertFalse(passive.is_open)
            TouchPoint(*button.to_window(*button.center))
            self.assertTrue(getattr(root, active.text[0::6] + 'button'))
            sleep(gdd.min_state_time)
            self.move_frames(1)
            self.assertNotEqual(gdd, self._win.children[0])
            self.assertLess(len(self._win.children), 2)
            self.assertFalse(active.is_open)
            self.assertFalse(passive.is_open)
            self.check_dropdown(present=False)
        self._win.remove_widget(root)

    def test_4_openmulti(self, *args):
        self._win = EventLoop.window
        self.clean_garbage()
        root = Builder.load_string(KV)
        self.render(root)
        self.assertLess(len(self._win.children), 2)
        group2 = root.ids.group2
        group2button = root.ids.group2button
        group1 = root.ids.group1
        group1button = root.ids.group1button
        self.move_frames(5)
        self.check_dropdown(present=False)
        self.assertFalse(group2.is_open)
        items = ((group2, group2button), (group1, group1button))
        for item in items:
            group, button = item
            for _ in range(5):
                TouchPoint(*group.center)
                self.check_dropdown(present=True)
                gdd = WeakProxy(self._win.children[0])
                self.assertIn(gdd, self._win.children)
                self.assertEqual(gdd, self._win.children[0])
                self.assertTrue(group.is_open)
                TouchPoint(*button.to_window(*button.center))
                sleep(gdd.min_state_time)
                self.move_frames(1)
                self.assertNotEqual(gdd, self._win.children[0])
                self.assertFalse(group.is_open)
                self.check_dropdown(present=False)
        self._win.remove_widget(root)