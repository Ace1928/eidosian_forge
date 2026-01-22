import os
from glob import glob
from shutil import rmtree
from string import Template
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import simplify_list, ensure_list
from ...utils.misc import human_order_sorted
from ...external.due import BibTeX
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
def _get_pe_files(self, cwd):
    files = None
    if isdefined(self.inputs.design_file):
        fp = open(self.inputs.design_file, 'rt')
        for line in fp.readlines():
            if line.startswith('/NumWaves'):
                numpes = int(line.split()[-1])
                files = []
                for i in range(numpes):
                    files.append(self._gen_fname('pe%d.nii' % (i + 1), cwd=cwd))
                break
        fp.close()
    return files