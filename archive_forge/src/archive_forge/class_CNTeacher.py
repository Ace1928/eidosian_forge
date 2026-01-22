from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class CNTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt['datafile'] = _path('cbtest_CN', opt)
        opt['cloze'] = True
        super().__init__(opt, shared)