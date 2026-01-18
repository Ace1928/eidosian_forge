import numpy
import pytest
from thinc.api import Optimizer, registry
def test_optimizer_schedules_valid(schedule_valid):
    lr, lr_next1, lr_next2, lr_next3 = schedule_valid
    optimizer = Optimizer(learn_rate=lr)
    assert optimizer.learn_rate == lr_next1
    optimizer.step_schedules()
    assert optimizer.learn_rate == lr_next2
    optimizer.step_schedules()
    assert optimizer.learn_rate == lr_next3
    optimizer.learn_rate = 1.0
    assert optimizer.learn_rate == 1.0