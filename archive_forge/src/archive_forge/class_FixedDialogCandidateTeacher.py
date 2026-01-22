from parlai.core.teachers import (
from parlai.core.opt import Opt
import copy
import random
import itertools
import os
from PIL import Image
import string
import json
from abc import ABC
from typing import Tuple, List
class FixedDialogCandidateTeacher(CandidateBaseTeacher, FixedDialogTeacher):
    """
    Base Candidate Teacher.

    Useful if you'd like to test the FixedDialogTeacher
    """

    def __init__(self, *args, **kwargs):
        """
        Override to build candidates.
        """
        super().__init__(*args, **kwargs)
        opt = args[0]
        if 'shared' not in kwargs:
            self._setup_data(opt['datatype'].split(':')[0])
            self._build_candidates()
        else:
            shared = kwargs['shared']
            self.corpus = shared['corpus']
            self.cands = shared['cands']
        self.reset()

    def share(self):
        shared = super().share()
        shared['corpus'] = self.corpus
        shared['cands'] = self.cands
        return shared

    def _build_candidates(self):
        self.cands = []
        for i in range(len(self.corpus)):
            cands = []
            for j in range(NUM_CANDIDATES):
                offset = (i + j) % len(self.corpus)
                cands.append(self.corpus[offset])
            self.cands.append(cands)

    def get(self, episode_idx: int, entry_idx: int=0):
        return {'text': self.corpus[episode_idx], 'episode_done': True, 'labels': [self.corpus[episode_idx]], 'label_candidates': self.cands[episode_idx]}