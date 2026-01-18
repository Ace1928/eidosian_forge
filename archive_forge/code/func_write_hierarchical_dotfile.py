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
def write_hierarchical_dotfile(self, dotfilename=None, colored=False, simple_form=True):
    dotlist = ['digraph %s{' % self.name]
    dotlist.append(self._get_dot(prefix='  ', colored=colored, simple_form=simple_form))
    dotlist.append('}')
    dotstr = '\n'.join(dotlist)
    if dotfilename:
        fp = open(dotfilename, 'wt')
        fp.writelines(dotstr)
        fp.close()
    else:
        logger.info(dotstr)