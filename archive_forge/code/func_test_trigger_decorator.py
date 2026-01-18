import gc
import weakref
import pytest
def test_trigger_decorator(kivy_clock, clock_counter):
    from kivy.clock import triggered

    @triggered()
    def triggered_callback():
        clock_counter(dt=0)
    triggered_callback()
    assert clock_counter.counter == 0
    kivy_clock.tick()
    assert clock_counter.counter == 1