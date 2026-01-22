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
class RepeatWordsTeacher(NocandidateTeacher):
    """
    Each input/output pair is a word repeated n times.

    Useful for testing beam-blocking.
    """

    def __init__(self, *args, **kwargs):
        kwargs['vocab_size'] = 70
        kwargs['example_size'] = 11
        super().__init__(*args, **kwargs)

    def build_corpus(self):
        """
        Override to repeat words.
        """
        return [[x for _ in range(l)] for l in range(1, self.example_size) for x in self.words]