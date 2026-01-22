import linecache
import reprlib
import traceback
from . import base_futures
from . import coroutines
def _task_repr_info(task):
    info = base_futures._future_repr_info(task)
    if task.cancelling() and (not task.done()):
        info[0] = 'cancelling'
    info.insert(1, 'name=%r' % task.get_name())
    coro = coroutines._format_coroutine(task._coro)
    info.insert(2, f'coro=<{coro}>')
    if task._fut_waiter is not None:
        info.insert(3, f'wait_for={task._fut_waiter!r}')
    return info