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
def _forward_event(self, msg):
    if msg['event'] == 'initialized':
        self.init_event.set()
        self.init_event_seq = msg['seq']
    self.event_callback(msg)