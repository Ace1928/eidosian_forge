import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.mark.parametrize('n', [1, 10, 100, 1000])
def test_widget_empty_draw(kivy_benchmark, n):
    from kivy.graphics import RenderContext
    from kivy.uix.widget import Widget
    ctx = RenderContext()
    root = Widget()
    for x in range(n):
        root.add_widget(Widget())
    ctx.add(root.canvas)
    kivy_benchmark(ctx.draw)