from __future__ import annotations
import os
import traceback
from typing import TYPE_CHECKING, Final, cast
import streamlit
from streamlit.errors import (
from streamlit.logger import get_logger
from streamlit.proto.Exception_pb2 import Exception as ExceptionProto
from streamlit.runtime.metrics_util import gather_metrics
def _get_stack_trace_str_list(exception: BaseException, strip_streamlit_stack_entries: bool=False) -> list[str]:
    """Get the stack trace for the given exception.

    Parameters
    ----------
    exception : BaseException
        The exception to extract the traceback from

    strip_streamlit_stack_entries : bool
        If True, all traceback entries that are in the Streamlit package
        will be removed from the list. We do this for exceptions that result
        from incorrect usage of Streamlit APIs, so that the user doesn't see
        a bunch of noise about ScriptRunner, DeltaGenerator, etc.

    Returns
    -------
    list
        The exception traceback as a list of strings

    """
    extracted_traceback: traceback.StackSummary | None = None
    if isinstance(exception, StreamlitAPIWarning):
        extracted_traceback = exception.tacked_on_stack
    elif hasattr(exception, '__traceback__'):
        extracted_traceback = traceback.extract_tb(exception.__traceback__)
    if isinstance(exception, UncaughtAppException):
        extracted_traceback = traceback.extract_tb(exception.exc.__traceback__)
    if extracted_traceback is None:
        stack_trace_str_list = ['Cannot extract the stack trace for this exception. Try calling exception() within the `catch` block.']
    elif strip_streamlit_stack_entries:
        extracted_frames = _get_nonstreamlit_traceback(extracted_traceback)
        stack_trace_str_list = traceback.format_list(extracted_frames)
    else:
        stack_trace_str_list = traceback.format_list(extracted_traceback)
    stack_trace_str_list = [item.strip() for item in stack_trace_str_list]
    return stack_trace_str_list