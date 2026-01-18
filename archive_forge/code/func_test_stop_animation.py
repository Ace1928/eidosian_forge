import pytest
def test_stop_animation(self):
    from kivy.animation import Animation
    from kivy.uix.widget import Widget
    a = Animation(x=100, d=1)
    w = Widget()
    a.start(w)
    sleep(0.5)
    a.stop(w)
    assert w.x != pytest.approx(100)
    assert w.x != pytest.approx(0)
    assert no_animations_being_played()