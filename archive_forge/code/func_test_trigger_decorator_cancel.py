import gc
import weakref
import pytest
def test_trigger_decorator_cancel(kivy_clock, clock_counter):
    from kivy.clock import triggered

    @triggered()
    def triggered_callback():
        clock_counter(dt=0)
    triggered_callback()
    triggered_callback.cancel()
    kivy_clock.tick()
    assert clock_counter.counter == 0