import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def _threaded_measurement_loop(self, number: int, time_hook: Callable[[], float], stop_hook: Callable[[List[float]], bool], min_run_time: float, max_run_time: Optional[float]=None, callback: Optional[Callable[[int, float], NoReturn]]=None) -> List[float]:
    total_time = 0.0
    can_stop = False
    times: List[float] = []
    with common.set_torch_threads(self._task_spec.num_threads):
        while total_time < min_run_time or not can_stop:
            time_spent = time_hook()
            times.append(time_spent)
            total_time += time_spent
            if callback:
                callback(number, time_spent)
            can_stop = stop_hook(times)
            if max_run_time and total_time > max_run_time:
                break
    return times