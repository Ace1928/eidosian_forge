import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def _break_traceback_cycle_from_frame(frame):
    this_frame = sys._getframe(0)
    refs = gc.get_referrers(frame)
    while refs:
        for frame in refs:
            if frame is not this_frame and isinstance(frame, types.FrameType):
                break
        else:
            break
        refs = None
        frame.clear()
        refs = gc.get_referrers(frame)
    refs = frame = this_frame = None