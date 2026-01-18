import gc
import weakref
import pytest
def test_schedule_once_twice(kivy_clock, clock_counter):
    kivy_clock.schedule_once(clock_counter)
    kivy_clock.schedule_once(clock_counter)
    kivy_clock.tick()
    assert clock_counter.counter == 2