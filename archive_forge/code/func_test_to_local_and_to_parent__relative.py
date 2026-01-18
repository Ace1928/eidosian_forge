import pytest
import functools
@pytest.mark.parametrize('widget_cls_name', relative_type_widget_cls_names)
def test_to_local_and_to_parent__relative(widget_cls_name, kivy_clock):
    from kivy.factory import Factory
    widget = Factory.get(widget_cls_name)(pos=(100, 100))
    kivy_clock.tick()
    assert widget.to_local(0, 0) == (-100, -100)
    assert widget.to_parent(0, 0) == (100, 100)