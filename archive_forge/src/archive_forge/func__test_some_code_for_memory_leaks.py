from collections import defaultdict, namedtuple
import gc
import os
import re
import time
import tracemalloc
from typing import Callable, List, Optional
from ray.util.annotations import DeveloperAPI
def _test_some_code_for_memory_leaks(desc: str, init: Optional[Callable[[], None]], code: Callable[[], None], repeats: int, max_num_trials: int=1) -> List[Suspect]:
    """Runs given code (and init code) n times and checks for memory leaks.

    Args:
        desc: A descriptor of the test.
        init: Optional code to be executed initially.
        code: The actual code to be checked for producing memory leaks.
        repeats: How many times to repeatedly execute `code`.
        max_num_trials: The maximum number of trials to run. A new trial is only
            run, if the previous one produced a memory leak. For all non-1st trials,
            `repeats` calculates as: actual_repeats = `repeats` * (trial + 1), where
            the first trial is 0.

    Returns:
        A list of Suspect objects, describing possible memory leaks. If list
        is empty, no leaks have been found.
    """

    def _i_print(i):
        if (i + 1) % 10 == 0:
            print('.', end='' if (i + 1) % 100 else f' {i + 1}\n', flush=True)
    suspicious = set()
    suspicious_stats = []
    for trial in range(max_num_trials):
        tracemalloc.start(20)
        table = defaultdict(list)
        actual_repeats = repeats * (trial + 1)
        print(f'{desc} {actual_repeats} times.')
        if init is not None:
            init()
        for i in range(actual_repeats):
            _i_print(i)
            gc.collect()
            code()
            gc.collect()
            _take_snapshot(table, suspicious)
        print('\n')
        suspicious.clear()
        suspicious_stats.clear()
        suspects = _find_memory_leaks_in_table(table)
        for suspect in sorted(suspects, key=lambda s: s.memory_increase, reverse=True):
            if len(suspicious) == 0:
                _pprint_suspect(suspect)
                print('-> added to retry list')
            suspicious.add(suspect.traceback)
            suspicious_stats.append(suspect)
        tracemalloc.stop()
        if len(suspicious) > 0:
            print(f'{len(suspicious)} suspects found. Top-ten:')
            for i, s in enumerate(suspicious_stats):
                if i > 10:
                    break
                print(f'{i}) line={s.traceback[-1]} mem-increase={s.memory_increase}B slope={s.slope}B/detection rval={s.rvalue}')
        else:
            print('No remaining suspects found -> returning')
            break
    if len(suspicious_stats) > 0:
        _pprint_suspect(suspicious_stats[0])
    return suspicious_stats