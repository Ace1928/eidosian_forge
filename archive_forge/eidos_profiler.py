import logging
from typing import Callable
import sys


def _trace_function(
    frame, event, arg, logger: logging.Logger, trace_level: int
) -> Callable:
    """⚙️ Implements detailed tracing of function calls."""
    if event == "call":
        code = frame.f_code
        func_name = code.co_name
        file_name = code.co_filename
        line_no = frame.f_lineno
        logger.log(
            trace_level,
            f"🔍 TRACE: Calling function '{func_name}' in '{file_name}:{line_no}'",
        )
    elif event == "return":
        code = frame.f_code
        func_name = code.co_name
        file_name = code.co_filename
        line_no = frame.f_lineno
        logger.log(
            trace_level,
            f"🔍 TRACE: Returning from function '{func_name}' in '{file_name}:{line_no}'",
        )
    return _trace_function
