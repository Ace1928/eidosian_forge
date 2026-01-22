import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from .build import build
import json
import os
import random
class BasicBothDialogTeacher(MultiTaskTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = 'wizard_of_wikipedia:BasicApprenticeDialog,wizard_of_wikipedia:BasicWizardDialog'
        super().__init__(opt, shared)