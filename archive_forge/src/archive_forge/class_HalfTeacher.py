from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018
import copy
import os
class HalfTeacher(FbDialogTeacher):
    """
    This version of opensubtitles creates half of all possible dialog examples.
    """

    def __init__(self, opt, shared=None, version='2018', use_history=True):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, version, use_history)
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

    def setup_data(self, path):
        for entry, new in super().setup_data(path):
            if entry[1]:
                yield (entry, new)