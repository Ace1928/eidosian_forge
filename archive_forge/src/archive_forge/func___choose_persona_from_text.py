import copy
import json
import os
import random
import re
from collections import defaultdict
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from parlai.core.opt import Opt
from parlai.core.teachers import (
from parlai.tasks.convai2.agents import (
from parlai.tasks.empathetic_dialogues.agents import EmpatheticDialoguesTeacher
from parlai.tasks.wizard_of_wikipedia.agents import WizardDialogKnowledgeTeacher
from parlai.utils.misc import warn_once
from .build import build
def __choose_persona_from_text(self, utt):
    utt = utt.strip()
    if utt not in self.utterance_to_persona_map:
        best_word_overlap = 0
        best_persona = None
        for p in self.personas:
            word_overlap = self.__calculate_word_overlap(utt, p)
            if word_overlap >= best_word_overlap:
                best_word_overlap = word_overlap
                best_persona = p
        if not best_persona:
            raise Exception(f'No persona found for utterance: "{utt}". This should not happen.')
        self.utterance_to_persona_map[utt] = best_persona
        return best_persona
    return self.utterance_to_persona_map[utt]