import os
from typing import Any, List
import numpy as np
from parlai.core.teachers import FixedDialogTeacher
from .build import build
class EmotionClassificationSituationTeacher(EmpatheticDialoguesTeacher):
    """
    Class for detecting the emotion based on the situation.
    """

    def __init__(self, opt, shared=None):
        opt['train_experiencer_only'] = True
        super().__init__(opt, shared)
        if not shared:
            self._get_situations()

    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return len(self.data)

    def _get_situations(self):
        new_data = []
        for ep in self.data:
            new_data.append(ep[0])
        self.data = new_data

    def get(self, episode_idx, entry_idx=0):
        ex = self.data[episode_idx]
        episode_done = True
        return {'labels': [ex[2]], 'text': ex[3], 'episode_done': episode_done}