import functools
import inspect
import opcode
import os
import sys
import traceback
import types
from . import events
from . import futures
from .log import logger
def yield_from_gen(gen):
    yield from gen