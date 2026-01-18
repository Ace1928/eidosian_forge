import gc
import weakref
import pytest
@triggered()
def triggered_callback():
    clock_counter(dt=0)