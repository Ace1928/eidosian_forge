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
def _create_ev_file(self, evfname, evinfo):
    f = open(evfname, 'wt')
    for i in evinfo:
        if len(i) == 3:
            f.write('%f %f %f\n' % (i[0], i[1], i[2]))
        else:
            f.write('%f\n' % i[0])
    f.close()