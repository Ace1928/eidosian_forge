import gc
import weakref
import pytest
def test_schedule_once_draw_before(kivy_clock, clock_counter):
    kivy_clock.schedule_once(clock_counter, -1)
    kivy_clock.tick_draw()
    assert clock_counter.counter == 1
    kivy_clock.tick()
    assert clock_counter.counter == 1