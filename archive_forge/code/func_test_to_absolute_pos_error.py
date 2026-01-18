import pytest
from kivy.compat import isclose
from kivy.input import MotionEvent
def test_to_absolute_pos_error(self):
    event = self.create_dummy_motion_event()
    with pytest.raises(ValueError):
        event.to_absolute_pos(0, 0, 100, 100, 10)