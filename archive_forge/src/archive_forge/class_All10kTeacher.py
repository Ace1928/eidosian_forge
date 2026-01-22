from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class All10kTeacher(MultiTaskTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join(('babi:Task10k:%d' % (i + 1) for i in range(20)))
        super().__init__(opt, shared)