import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class BasicApprenticeDialogTeacher(BasicdialogTeacher):

    def __init__(self, opt, shared=None):
        opt['speaker_label'] = 'apprentice'
        super().__init__(opt, shared)