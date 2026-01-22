import io
import sys
import traceback
from . import util
from functools import wraps
def _is_relevant_tb_level(self, tb):
    return '__unittest' in tb.tb_frame.f_globals