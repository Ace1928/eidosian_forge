import os
import math
import sys
from typing import Optional, Union, Callable
import pyglet
from pyglet.customtypes import Buffer
def _debug_print_dummy(arg: str) -> bool:
    return True