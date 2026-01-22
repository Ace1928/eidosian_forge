import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _write_inputs(node):
    lines = []
    nodename = node.fullname.replace('.', '_')
    for key, _ in list(node.inputs.items()):
        val = getattr(node.inputs, key)
        if isdefined(val):
            if isinstance(val, (str, bytes)):
                try:
                    func = create_function_from_source(val)
                except RuntimeError:
                    lines.append("%s.inputs.%s = '%s'" % (nodename, key, val))
                else:
                    funcname = [name for name in func.__globals__ if name != '__builtins__'][0]
                    lines.append(pickle.loads(val))
                    if funcname == nodename:
                        lines[-1] = lines[-1].replace(' %s(' % funcname, ' %s_1(' % funcname)
                        funcname = '%s_1' % funcname
                    lines.append('from nipype.utils.functions import getsource')
                    lines.append('%s.inputs.%s = getsource(%s)' % (nodename, key, funcname))
            else:
                lines.append('%s.inputs.%s = %s' % (nodename, key, val))
    return lines