import gc
import weakref
import pytest
def test_trigger_interval(kivy_clock, clock_counter):
    trigger = kivy_clock.create_trigger(clock_counter, 0, interval=True)
    trigger()
    kivy_clock.tick()
    assert clock_counter.counter == 1
    kivy_clock.tick()
    assert clock_counter.counter == 2