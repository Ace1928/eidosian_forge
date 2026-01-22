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
def _get_all_nodes(self):
    allnodes = self._nodes_cache - self._nested_workflows_cache
    for node in self._nested_workflows_cache:
        allnodes |= node._get_all_nodes()
    return allnodes