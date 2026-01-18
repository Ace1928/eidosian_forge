import gc
import weakref
import pytest
def test_unschedule_event(kivy_clock, clock_counter):
    ev = kivy_clock.schedule_once(clock_counter)
    kivy_clock.unschedule(ev)
    kivy_clock.tick()
    assert clock_counter.counter == 0