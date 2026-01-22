import pytest
class TestSequence:

    def test_cancel_all(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.cancel_all(w)
        assert no_animations_being_played()

    def test_cancel_all_2(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.cancel_all(w, 'x')
        assert no_animations_being_played()

    def test_stop_all(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w)
        assert no_animations_being_played()

    def test_stop_all_2(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w, 'x')
        assert no_animations_being_played()

    def test_count_events(self, ec_cls):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0.5) + Animation(x=0, d=0.5)
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(0, False, 0)
        sleep(0.2)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(0, False, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, True, 1)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 1)
        assert no_animations_being_played()

    def test_have_properties_to_animate(self):
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        assert not a.have_properties_to_animate(w)
        a.start(w)
        assert a.have_properties_to_animate(w)
        a.stop(w)
        assert not a.have_properties_to_animate(w)
        assert no_animations_being_played()

    def test_animated_properties(self):
        from kivy.animation import Animation
        a = Animation(x=100, y=200) + Animation(x=0)
        assert a.animated_properties == {'x': 0, 'y': 200}

    def test_transition(self):
        from kivy.animation import Animation
        a = Animation(x=100) + Animation(x=0)
        with pytest.raises(AttributeError):
            a.transition