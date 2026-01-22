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
def _make_nodes(self, cwd=None):
    if cwd is None:
        cwd = self.output_dir()
    if self.nested:
        nitems = len(flatten(ensure_list(getattr(self.inputs, self.iterfield[0]))))
    else:
        nitems = len(ensure_list(getattr(self.inputs, self.iterfield[0])))
    for i in range(nitems):
        nodename = '_%s%d' % (self.name, i)
        node = Node(deepcopy(self._interface), n_procs=self._n_procs, mem_gb=self._mem_gb, overwrite=self.overwrite, needed_outputs=self.needed_outputs, run_without_submitting=self.run_without_submitting, base_dir=op.join(cwd, 'mapflow'), name=nodename)
        node.plugin_args = self.plugin_args
        node.interface.inputs.trait_set(**deepcopy(self._interface.inputs.trait_get()))
        node.interface.resource_monitor = self._interface.resource_monitor
        for field in self.iterfield:
            if self.nested:
                fieldvals = flatten(ensure_list(getattr(self.inputs, field)))
            else:
                fieldvals = ensure_list(getattr(self.inputs, field))
            logger.debug('setting input %d %s %s', i, field, fieldvals[i])
            setattr(node.inputs, field, fieldvals[i])
        node.config = self.config
        yield (i, node)