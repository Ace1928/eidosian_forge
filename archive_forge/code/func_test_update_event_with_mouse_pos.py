from kivy.tests.common import GraphicUnitTest
def test_update_event_with_mouse_pos(self):
    win, mouse = self.get_providers()
    win.mouse_pos = (10.0, 10.0)
    x, y = win.mouse_pos = (50.0, 50.0)
    self.advance_frames(1)
    self.assert_event('update', win.to_normalized_pos(x, y))
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)