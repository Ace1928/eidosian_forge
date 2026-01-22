import copy
import json
import os
import re
import numpy as np
from parlai.core.teachers import ParlAIDialogTeacher, MultiTaskTeacher
from projects.self_feeding.utils import add_person_tokens
from .build import build
class DialogTeacher(SelfFeedingTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['subtask'] = 'dialog'
        super().__init__(opt, shared)

    @staticmethod
    def add_cmdline_args(argparser):
        SelfFeedingTeacher.add_cmdline_args(argparser)