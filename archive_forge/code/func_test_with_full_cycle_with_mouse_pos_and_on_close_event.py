from kivy.tests.common import GraphicUnitTest
def test_with_full_cycle_with_mouse_pos_and_on_close_event(self):
    win, mouse = self.get_providers()
    x, y = win.mouse_pos = (5.0, 5.0)
    self.advance_frames(1)
    self.assert_event('begin', win.to_normalized_pos(x, y))
    x, y = win.mouse_pos = (10.0, 10.0)
    self.advance_frames(1)
    self.assert_event('update', win.to_normalized_pos(x, y))
    win.dispatch('on_close')
    self.advance_frames(1)
    self.assert_event('end', win.to_normalized_pos(x, y))