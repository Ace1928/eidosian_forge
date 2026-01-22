import enum
import timeit
import textwrap
from typing import overload, Any, Callable, Dict, List, NoReturn, Optional, Tuple, Type, Union
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
class CPPTimer:

    def __init__(self, stmt: str, setup: str, global_setup: str, timer: Callable[[], float], globals: Dict[str, Any]) -> None:
        if timer is not timeit.default_timer:
            raise NotImplementedError("PyTorch was built with CUDA and a GPU is present; however Timer does not yet support GPU measurements. If your code is CPU only, pass `timer=timeit.default_timer` to the Timer's constructor to indicate this. (Note that this will produce incorrect results if the GPU is in fact used, as Timer will not synchronize CUDA.)")
        if globals:
            raise ValueError('C++ timing does not support globals.')
        self._stmt: str = textwrap.dedent(stmt)
        self._setup: str = textwrap.dedent(setup)
        self._global_setup: str = textwrap.dedent(global_setup)
        self._timeit_module: Optional[TimeitModuleType] = None

    def timeit(self, number: int) -> float:
        if self._timeit_module is None:
            self._timeit_module = cpp_jit.compile_timeit_template(stmt=self._stmt, setup=self._setup, global_setup=self._global_setup)
        return self._timeit_module.timeit(number)