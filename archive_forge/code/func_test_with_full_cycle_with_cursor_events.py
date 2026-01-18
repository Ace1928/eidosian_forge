from kivy.tests.common import GraphicUnitTest
def test_with_full_cycle_with_cursor_events(self):
    win, mouse = self.get_providers()
    win.dispatch('on_cursor_enter')
    x, y = win.mouse_pos
    self.advance_frames(1)
    self.assert_event('begin', win.to_normalized_pos(x, y))
    x, y = win.mouse_pos = (10.0, 10.0)
    self.advance_frames(1)
    self.assert_event('update', win.to_normalized_pos(x, y))
    win.dispatch('on_cursor_leave')
    self.advance_frames(1)
    self.assert_event('end', win.to_normalized_pos(x, y))