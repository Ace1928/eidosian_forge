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
def _get_example(self, episode_idx: int, entry_idx: Optional[int]=None):
    """
        Get example from the base ED teacher and add persona and WoW topic strings.
        """
    gotten = super().get(episode_idx, entry_idx=entry_idx)
    if entry_idx == 0:
        modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
        gotten['text'] = modified_text
    return gotten