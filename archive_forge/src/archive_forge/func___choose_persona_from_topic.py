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
def __choose_persona_from_topic(self, topic):
    topic = topic.strip()
    persona_strings = self.wow_topics_to_persona_strings_map[topic]
    for p in persona_strings:
        for persona in self.personas:
            if p in persona:
                return persona
    if self.no_persona_is_error:
        raise ValueError(f'ERROR: Found no persona for topic: {topic}.')
    else:
        warn_once(f'Found no persona for topic: {topic}. Returning first persona.')
        return self.personas[0]