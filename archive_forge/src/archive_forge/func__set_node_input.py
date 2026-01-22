import os
import os.path as op
import sys
from datetime import datetime
from copy import deepcopy
import pickle
import shutil
import numpy as np
from ... import config, logging
from ...utils.misc import str2bool
from ...utils.functions import getsource, create_function_from_source
from ...interfaces.base import traits, TraitedSpec, TraitDictObject, TraitListObject
from ...utils.filemanip import save_json
from .utils import (
from .base import EngineBase
from .nodes import MapNode
def _set_node_input(self, node, param, source, sourceinfo):
    """Set inputs of a node given the edge connection"""
    if isinstance(sourceinfo, (str, bytes)):
        val = source.get_output(sourceinfo)
    elif isinstance(sourceinfo, tuple):
        if callable(sourceinfo[1]):
            val = sourceinfo[1](source.get_output(sourceinfo[0]), *sourceinfo[2:])
    newval = val
    if isinstance(val, TraitDictObject):
        newval = dict(val)
    if isinstance(val, TraitListObject):
        newval = val[:]
    logger.debug('setting node input: %s->%s', param, str(newval))
    node.set_input(param, deepcopy(newval))