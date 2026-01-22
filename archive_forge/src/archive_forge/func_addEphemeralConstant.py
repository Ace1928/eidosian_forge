import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def addEphemeralConstant(self, name, ephemeral):
    """Add an ephemeral constant to the set."""
    PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)