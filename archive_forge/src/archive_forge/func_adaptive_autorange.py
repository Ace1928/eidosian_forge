import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
def adaptive_autorange(self, threshold: float=0.1, *, min_run_time: float=0.01, max_run_time: float=10.0, callback: Optional[Callable[[int, float], NoReturn]]=None) -> common.Measurement:
    """Similar to `blocked_autorange` but also checks for variablility in measurements
        and repeats until iqr/median is smaller than `threshold` or `max_run_time` is reached.


        At a high level, adaptive_autorange executes the following pseudo-code::

            `setup`

            times = []
            while times.sum < max_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                times.append(timer() - start)

                enough_data = len(times)>3 and times.sum > min_run_time
                small_iqr=times.iqr/times.mean<threshold

                if enough_data and small_iqr:
                    break

        Args:
            threshold: value of iqr/median threshold for stopping

            min_run_time: total runtime needed before checking `threshold`

            max_run_time: total runtime  for all measurements regardless of `threshold`

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
    number = self._estimate_block_size(min_run_time=0.05)

    def time_hook() -> float:
        return self._timeit(number)

    def stop_hook(times: List[float]) -> bool:
        if len(times) > 3:
            return common.Measurement(number_per_run=number, raw_times=times, task_spec=self._task_spec).meets_confidence(threshold=threshold)
        return False
    times = self._threaded_measurement_loop(number, time_hook, stop_hook, min_run_time, max_run_time, callback=callback)
    return common.Measurement(number_per_run=number, raw_times=times, task_spec=self._task_spec)