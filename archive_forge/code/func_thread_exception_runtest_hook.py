import threading
import traceback
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Type
import warnings
import pytest
def thread_exception_runtest_hook() -> Generator[None, None, None]:
    with catch_threading_exception() as cm:
        try:
            yield
        finally:
            if cm.args:
                thread_name = '<unknown>' if cm.args.thread is None else cm.args.thread.name
                msg = f'Exception in thread {thread_name}\n\n'
                msg += ''.join(traceback.format_exception(cm.args.exc_type, cm.args.exc_value, cm.args.exc_traceback))
                warnings.warn(pytest.PytestUnhandledThreadExceptionWarning(msg))