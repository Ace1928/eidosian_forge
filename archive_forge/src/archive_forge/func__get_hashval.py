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
def _get_hashval(self):
    """Compute hash including iterfield lists."""
    self._get_inputs()
    if self._hashvalue is not None and self._hashed_inputs is not None:
        return (self._hashed_inputs, self._hashvalue)
    self._check_iterfield()
    hashinputs = deepcopy(self._interface.inputs)
    for name in self.iterfield:
        hashinputs.remove_trait(name)
        hashinputs.add_trait(name, InputMultiPath(self._interface.inputs.traits()[name].trait_type))
        logger.debug('setting hashinput %s-> %s', name, getattr(self._inputs, name))
        if self.nested:
            setattr(hashinputs, name, flatten(getattr(self._inputs, name)))
        else:
            setattr(hashinputs, name, getattr(self._inputs, name))
    hashed_inputs, hashvalue = hashinputs.get_hashval(hash_method=self.config['execution']['hash_method'])
    rm_extra = self.config['execution']['remove_unnecessary_outputs']
    if str2bool(rm_extra) and self.needed_outputs:
        hashobject = md5()
        hashobject.update(hashvalue.encode())
        sorted_outputs = sorted(self.needed_outputs)
        hashobject.update(str(sorted_outputs).encode())
        hashvalue = hashobject.hexdigest()
        hashed_inputs.append(('needed_outputs', sorted_outputs))
    self._hashed_inputs, self._hashvalue = (hashed_inputs, hashvalue)
    return (self._hashed_inputs, self._hashvalue)