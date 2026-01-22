import os
from parlai.core.teachers import FbDialogTeacher
from parlai.tasks.dialog_babi_plus.build import build
class KBTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        build(opt)
        opt['datafile'] = os.path.join(opt['datapath'], 'dialog-bAbI-plus', 'dialog-bAbI-plus-tasks', 'dialog-babi-kb-all.txt')
        super().__init__(opt, shared)