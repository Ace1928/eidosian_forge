from kivy.tests.common import GraphicUnitTest
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.tests.common import UTMotionEvent
from time import sleep
from itertools import count
def test_scrollbar_both_margin(self):
    EventLoop.ensure_window()
    win = EventLoop.window
    grid = _TestGrid()
    scroll = _TestScrollbarBothMargin()
    margin = scroll.bar_margin
    scroll.add_widget(grid)
    win.add_widget(scroll)
    EventLoop.idle()
    left, right = scroll.to_window(scroll.x, scroll.right)
    bottom, top = scroll.to_window(scroll.y, scroll.top)
    m = margin + scroll.bar_width / 2.0
    points = [[left, bottom + m, right, bottom + m, 'bottom', 'right', False], [left, top - m, right, top - m, 'top', 'right', False], [right - m, top, right - m, bottom, 'bottom', 'right', False], [left + m, top, left + m, bottom, 'bottom', 'left', False], [left, bottom, right, bottom, 'bottom', 'right', True], [left, top, right, top, 'top', 'right', True], [right, top, right, bottom, 'bottom', 'right', True], [left, top, left, bottom, 'bottom', 'left', True]]
    self.process_points(scroll, points)
    self.render(scroll)