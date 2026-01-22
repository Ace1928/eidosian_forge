import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
def accept_variable(self, variable_name):
    """Accept a variable by name."""
    forbid_list = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__annotations__', '__builtins__', '__builtin__', '__display__', 'get_ipython', 'debugpy', 'exit', 'quit', 'In', 'Out', '_oh', '_dh', '_', '__', '___']
    cond = variable_name not in forbid_list
    cond = cond and (not bool(re.search('^_\\d', variable_name)))
    cond = cond and variable_name[0:2] != '_i'
    return cond