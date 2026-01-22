from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
class Task1kTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        default = '1'
        task = opt.get('task', f'babi:Task1k:{default}')
        self.task_num = task.split(':')[2] if len(task.split(':')) >= 3 else default
        opt['datafile'] = _path('', self.task_num, opt)
        opt['cands_datafile'] = _path('', self.task_num, opt, 'train')
        super().__init__(opt, shared)

    def setup_data(self, path):
        for entry, new in super().setup_data(path):
            entry[1] = mod_labels(entry[1], self.task_num)
            yield (entry, new)

    def load_cands(self, path):
        return mod_labels(super().load_cands(path), self.task_num)