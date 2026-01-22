import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
class _PdbWrap(Pdb):
    """Wrap PDB to run a custom exit hook on continue."""

    def __init__(self, exit_hook: Callable[[], None]):
        self._exit_hook = exit_hook
        Pdb.__init__(self)

    def do_continue(self, arg):
        self._exit_hook()
        return Pdb.do_continue(self, arg)
    do_c = do_cont = do_continue