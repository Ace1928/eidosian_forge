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
def _extract_personas(self, episode_idx: str) -> Tuple[List[str], List[str]]:
    """
        For the given ConvAI2 conversation, return strings of both speakers' personas.
        """
    first_entry = self.convai2_teacher.get(episode_idx, entry_idx=0)
    first_text_strings = first_entry['text'].split('\n')
    persona_1_strings = []
    persona_2_strings = []
    for str_ in first_text_strings[:-1]:
        if str_.startswith('your persona: '):
            persona_2_strings.append(str_[len('your persona: '):])
        elif str_.startswith("partner's persona: "):
            persona_1_strings.append(str_[len("partner's persona: "):])
        else:
            raise ValueError('Persona string cannot be parsed!')
    return (persona_1_strings, persona_2_strings)