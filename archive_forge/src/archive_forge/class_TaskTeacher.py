from parlai.core.teachers import FbDialogTeacher
from .build import build
import copy
import os
class TaskTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        params = opt['task'].split(':')[2]
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(os.path.join('babi', 'babi1'), params, opt)
        opt['cands_datafile'] = _path(os.path.join('babi', 'babi1'), params, opt, 'train')
        super().__init__(opt, shared)