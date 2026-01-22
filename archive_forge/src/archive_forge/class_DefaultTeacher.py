import copy
from .build import build, make_path
from parlai.utils.misc import warn_once
from parlai.core.teachers import ParlAIDialogTeacher
class DefaultTeacher(ParlAIDialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['parlaidialogteacher_datafile'] = _path(opt)
        super().__init__(opt, shared)