import pytest
@pytest.mark.parametrize(('n_slides', 'index', 'loop', 'index_of_previous_slide', 'index_of_next_slide'), ((1, 0, False, None, None), (1, 0, True, None, None), (2, 0, False, None, 1), (2, 0, True, 1, 1), (2, 1, False, 0, None), (2, 1, True, 0, 0), (3, 0, False, None, 1), (3, 0, True, 2, 1), (3, 1, False, 0, 2), (3, 1, True, 0, 2), (3, 2, False, 1, None), (3, 2, True, 1, 0)))
def test_previous_and_next(n_slides, index, loop, index_of_previous_slide, index_of_next_slide):
    from kivy.uix.carousel import Carousel
    from kivy.uix.widget import Widget
    c = Carousel(loop=loop)
    for i in range(n_slides):
        c.add_widget(Widget())
    c.index = index
    p_slide = c.previous_slide
    assert (p_slide and c.slides.index(p_slide)) == index_of_previous_slide
    n_slide = c.next_slide
    assert (n_slide and c.slides.index(n_slide)) == index_of_next_slide