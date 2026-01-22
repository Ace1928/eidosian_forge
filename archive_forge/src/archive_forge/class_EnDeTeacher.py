from parlai.core.teachers import DialogTeacher
from .build import build
import copy
import os
class EnDeTeacher(DefaultTeacher):

    def __init__(self, opt, shared=None):
        opt['task'] = 'iwslt14:en_de'
        super().__init__(opt, shared)