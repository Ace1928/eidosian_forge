from collections import OrderedDict, defaultdict
import os
import os.path as op
from pathlib import Path
import shutil
import socket
from copy import deepcopy
from glob import glob
from logging import INFO
from tempfile import mkdtemp
from ... import config, logging
from ...utils.misc import flatten, unflatten, str2bool, dict_diff
from ...utils.filemanip import (
from ...interfaces.base import (
from ...interfaces.base.specs import get_filecopy_info
from .utils import (
from .base import EngineBase
def _collate_results(self, nodes):
    finalresult = InterfaceResult(interface=[], runtime=[], provenance=[], inputs=[], outputs=self.outputs)
    returncode = []
    for i, nresult, err in nodes:
        finalresult.runtime.insert(i, None)
        returncode.insert(i, err)
        if nresult:
            if hasattr(nresult, 'runtime'):
                finalresult.interface.insert(i, nresult.interface)
                finalresult.inputs.insert(i, nresult.inputs)
                finalresult.runtime[i] = nresult.runtime
            if hasattr(nresult, 'provenance'):
                finalresult.provenance.insert(i, nresult.provenance)
        if self.outputs:
            for key, _ in list(self.outputs.items()):
                rm_extra = self.config['execution']['remove_unnecessary_outputs']
                if str2bool(rm_extra) and self.needed_outputs:
                    if key not in self.needed_outputs:
                        continue
                values = getattr(finalresult.outputs, key)
                if not isdefined(values):
                    values = []
                if nresult and nresult.outputs:
                    values.insert(i, nresult.outputs.trait_get()[key])
                else:
                    values.insert(i, None)
                defined_vals = [isdefined(val) for val in values]
                if any(defined_vals) and finalresult.outputs:
                    setattr(finalresult.outputs, key, values)
    if self.nested:
        for key, _ in list(self.outputs.items()):
            values = getattr(finalresult.outputs, key)
            if isdefined(values):
                values = unflatten(values, ensure_list(getattr(self.inputs, self.iterfield[0])))
            setattr(finalresult.outputs, key, values)
    if returncode and any([code is not None for code in returncode]):
        msg = []
        for i, code in enumerate(returncode):
            if code is not None:
                msg += ['Subnode %d failed' % i]
                msg += ['Error: %s' % str(code)]
        raise NodeExecutionError('Subnodes of node: %s failed:\n%s' % (self.name, '\n'.join(msg)))
    return finalresult