import gc
import weakref
import pytest
class ClockCounter:
    counter = 0

    def __call__(self, *args, **kwargs):
        self.counter += 1