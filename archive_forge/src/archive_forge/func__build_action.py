from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
def _build_action(self, entry):
    return {'text': self.get_text(entry), 'labels': [entry['label']], 'reward': 0, 'episode_done': True}