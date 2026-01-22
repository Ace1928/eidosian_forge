from parlai.core.teachers import DialogTeacher, MultiTaskTeacher
from .build import build
import copy
import json
import os
class VerifiedWikipediaTeacher(WikipediaTeacher):

    def __init__(self, opt, shared=None):
        self.prefix = 'verified-'
        self.suffix = 'dev'
        if opt['datatype'] != 'valid':
            print('WARNING: Verified teacher only provides dev data')
        opt['datafile'], self.evidence_dir = _path(opt)
        self.id = 'triviaqa'
        super().__init__(opt, shared)