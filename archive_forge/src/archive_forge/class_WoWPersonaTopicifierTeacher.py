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
class WoWPersonaTopicifierTeacher(WizardDialogKnowledgeTeacher):
    """
    Adds personas to WoW data.
    """

    def __init__(self, opt, shared=None):
        self.persona_topicifier = PersonaTopicifier(opt=opt, should_have_personas=False, should_have_topics=True)
        super().__init__(opt, shared=shared)

    def get(self, episode_idx, entry_idx=None):
        gotten = super().get(episode_idx, entry_idx=entry_idx)
        if entry_idx == 0:
            modified_text = self.persona_topicifier.get_modified_text(gotten['text'])
            gotten['text'] = modified_text
        return gotten