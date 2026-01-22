import os
from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.style_gen.build import (
class LabeledEDPersonaTopicifierTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:EDPersonaTopicifier, with Image-Chat personalities
    added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(opt=opt, base_task='blended_skill_talk:EDPersonaTopicifierTeacher')
        super().__init__(opt, shared=shared)