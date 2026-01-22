from __future__ import annotations
from typing import (
from simpy.exceptions import Interrupt
def _describe_frame(frame: FrameType) -> str:
    """Print filename, line number and function name of a stack frame."""
    filename, name = (frame.f_code.co_filename, frame.f_code.co_name)
    lineno = frame.f_lineno
    with open(filename) as f:
        for no, line in enumerate(f):
            if no + 1 == lineno:
                return f'  File "{filename}", line {lineno}, in {name}\n    {line.strip()}\n'
        return f'  File "{filename}", line {lineno}, in {name}\n'