import copy
import sys
import re
import os
from itertools import chain
from contextlib import contextmanager
from parso.python import tree
def values_from_qualified_names(inference_state, *names):
    return inference_state.import_module(names[:-1]).py__getattribute__(names[-1])