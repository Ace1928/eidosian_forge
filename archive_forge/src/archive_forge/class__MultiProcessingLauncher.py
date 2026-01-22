import itertools
import os
from dataclasses import dataclass
from multiprocessing.queues import SimpleQueue
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional
import torch
import torch.backends.cudnn
import torch.multiprocessing as mp
from lightning_utilities import apply_to_collection
from torch.nn import Module
from typing_extensions import override
from lightning_fabric.accelerators.cpu import CPUAccelerator
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.imports import _IS_INTERACTIVE
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
class _MultiProcessingLauncher(_Launcher):
    """Launches processes that run a given function in parallel, and joins them all at the end.

    The main process in which this launcher is invoked creates N so-called worker processes (using
    :func:`torch.multiprocessing.start_processes`) that run the given function.
    Worker processes have a rank that ranges from 0 to N - 1.

    Note:
        - This launcher requires all objects to be pickleable.
        - It is important that the entry point to the program/script is guarded by ``if __name__ == "__main__"``.
        - With start method 'fork' the user must ensure that no CUDA context gets created in the main process before
          the launcher is invoked. E.g., one should avoid creating cuda tensors or calling ``torch.cuda.*`` functions
          before calling ``Trainer.fit``.

    Args:
        strategy: A reference to the strategy that is used together with this launcher.
        start_method: The method how to start the processes.
            - 'spawn': The default start method. Requires all objects to be pickleable.
            - 'fork': Preferable for IPython/Jupyter environments where 'spawn' is not available. Not available on
              the Windows platform for example.
            - 'forkserver': Alternative implementation to 'fork'.

    """

    def __init__(self, strategy: 'ParallelStrategy', start_method: Literal['spawn', 'fork', 'forkserver']='spawn') -> None:
        self._strategy = strategy
        self._start_method = start_method
        if start_method not in mp.get_all_start_methods():
            raise ValueError(f"The start method '{self._start_method}' is not available on this platform. Available methods are: {', '.join(mp.get_all_start_methods())}")

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return self._start_method == 'fork'

    @override
    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Launches processes that run the given function in parallel.

        The function is allowed to have a return value. However, when all processes join, only the return value
        of worker process 0 gets returned from this `launch` method in the main process.

        Arguments:
            function: The entry point for all launched processes.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        if self._start_method in ('fork', 'forkserver'):
            _check_bad_cuda_fork()
        if self._start_method == 'spawn':
            _check_missing_main_guard()
        assert self._strategy.cluster_environment is not None
        os.environ['MASTER_PORT'] = str(self._strategy.cluster_environment.main_port)
        context = mp.get_context(self._start_method)
        return_queue = context.SimpleQueue()
        if self._start_method == 'spawn':
            global_states = _GlobalStateSnapshot.capture()
            process_args = [function, args, kwargs, return_queue, global_states]
        else:
            process_args = [function, args, kwargs, return_queue]
        mp.start_processes(self._wrapping_function, args=process_args, nprocs=self._strategy.num_processes, start_method=self._start_method)
        return return_queue.get()

    def _wrapping_function(self, process_idx: int, function: Callable, args: Any, kwargs: Any, return_queue: SimpleQueue, global_states: Optional['_GlobalStateSnapshot']=None) -> None:
        if global_states:
            global_states.restore()
        if self._start_method == 'spawn' and isinstance(self._strategy.accelerator, CPUAccelerator):
            args, kwargs = _disable_module_memory_sharing((args, kwargs))
        _set_num_threads_if_needed(num_processes=self._strategy.num_processes)
        os.environ['LOCAL_RANK'] = str(process_idx)
        results = function(*args, **kwargs)
        if process_idx == 0:
            return_queue.put(move_data_to_device(results, 'cpu'))