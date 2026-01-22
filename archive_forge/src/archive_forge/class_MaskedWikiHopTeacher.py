from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
class MaskedWikiHopTeacher(DefaultTeacher):

    def _path(self, opt):
        dt = opt['datatype'].split(':')
        datatype = 'train.masked' if dt[0] == 'train' else 'dev.masked'
        return os.path.join(opt['datapath'], 'qangaroo', 'qangaroo_v1.1', 'wikihop', datatype + '.json')