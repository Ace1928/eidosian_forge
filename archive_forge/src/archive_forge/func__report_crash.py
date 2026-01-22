import sys
from copy import deepcopy
from glob import glob
import os
import shutil
from time import sleep, time
from traceback import format_exception
import numpy as np
from ... import logging
from ...utils.misc import str2bool
from ..engine.utils import topological_sort, load_resultfile
from ..engine import MapNode
from .tools import report_crash, report_nodes_not_run, create_pyscript
def _report_crash(self, node, result=None):
    tb = None
    if result is not None:
        node._result = result['result']
        tb = result['traceback']
        node._traceback = tb
    return report_crash(node, traceback=tb)