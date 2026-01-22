from parlai.core.agents import create_agent_from_shared
from parlai.mturk.core.legacy_2018.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.legacy_2018.worlds import MTurkOnboardWorld
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply
from joblib import Parallel, delayed
import numpy as np
import os
import json
import random
import time
import torch
import copy
class PersonasGenerator(object):

    def __init__(self, opt):
        self.text_file = self._path(opt)
        self.personas = self.extract_personas()

    def _path(self, opt):
        persona = opt['persona_type']
        datatype = opt['persona_datatype'].split(':')[0]
        dt = datatype + '_' + persona
        if datatype == 'test':
            return os.path.join(opt['parlai_home'], 'parlai_internal/projects/convai2/test_set', dt + '_original_no_cands.txt')
        return os.path.join(opt['datapath'], 'ConvAI2', dt + '_original_no_cands.txt')

    def extract_personas(self):
        personas = []
        with open(self.text_file, 'r') as f:
            lines = f.readlines()
        new_persona = []
        for line in lines:
            if 'persona: ' in line:
                new_persona.append(line.split('persona: ')[1].replace('\n', ''))
            elif new_persona:
                personas.append(new_persona)
                new_persona = []
        return personas

    def get_persona(self):
        return random.choice(self.personas)