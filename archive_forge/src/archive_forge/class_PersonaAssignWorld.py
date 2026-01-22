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
class PersonaAssignWorld(MTurkOnboardWorld):
    """
    A world that assigns a persona to an agent.
    """

    def __init__(self, opt, mturk_agent):
        self.max_persona_time = opt['max_persona_time']
        self.human_eval = opt['human_eval']
        super().__init__(opt, mturk_agent)

    def parley(self):
        personas = self.mturk_agent.personas_generator.get_persona()
        self.mturk_agent.personas = personas
        if not self.human_eval:
            model_personas = self.mturk_agent.personas_generator.get_persona()
            while model_personas == personas:
                model_personas = self.mturk_agent.personas_generator.get_persona()
            self.mturk_agent.model_personas = model_personas
        persona_text = ''
        for persona in personas:
            persona_text += '<b><span style="color:blue">{}\n</span></b>'.format(persona.strip())
        self.mturk_agent.observe({'id': 'SYSTEM', 'show_persona': True, 'text': ONBOARD_MSG + '<br>' + persona_text + '<br>'})
        act = self.mturk_agent.act(timeout=self.max_persona_time)
        timed_out = self.check_timeout(act)
        if timed_out:
            self.episodeDone = True
            return

    def check_timeout(self, act):
        if 'text' in act:
            if act['text'] == '[TIMEOUT]' or act['text'] == '[RETURNED]' or act['text'] == '[DISCONNECT]':
                return True
        return False