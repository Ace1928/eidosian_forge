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
def _build_variables_response(self, request, variables):
    var_list = [var for var in variables if self.accept_variable(var['name'])]
    return {'seq': request['seq'], 'type': 'response', 'request_seq': request['seq'], 'success': True, 'command': request['command'], 'body': {'variables': var_list}}