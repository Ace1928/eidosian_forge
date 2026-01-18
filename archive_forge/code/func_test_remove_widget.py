import pytest
def test_remove_widget(loop):
    from kivy.uix.carousel import Carousel
    from kivy.uix.widget import Widget
    c = Carousel(loop=loop)
    assert c.index is None
    assert c.current_slide is None
    assert len(c.children) == 0
    assert len(c.slides) == 0
    N_SLIDES = 4
    for i in range(N_SLIDES):
        c.add_widget(Widget())
    assert c.index == 0
    assert c.current_slide == c.slides[0]
    assert len(c.children) == 3 if loop else 2
    assert len(c.slides) == N_SLIDES
    c.index = len(c.slides) - 1
    c.remove_widget(c.slides[0])
    c.index = 1
    old_current_slide = c.current_slide
    c.remove_widget(c.slides[c.index - 1])
    assert c.index == 0
    assert c.current_slide is old_current_slide
    assert len(c.children) == 2
    assert len(c.slides) == 2
    old_current_slide = c.current_slide
    c.remove_widget(c.slides[c.index + 1])
    assert c.index == 0
    assert c.current_slide is old_current_slide
    assert len(c.children) == 1
    assert len(c.slides) == 1
    c.remove_widget(c.current_slide)
    assert c.index is None
    assert c.current_slide is None
    assert len(c.children) == 0
    assert len(c.slides) == 0