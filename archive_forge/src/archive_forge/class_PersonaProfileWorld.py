from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from joblib import Parallel, delayed
from extract_and_save_personas import main as main_extract
import numpy as np
import time
import os
import pickle
import random
class PersonaProfileWorld(MTurkOnboardWorld):
    """
    A world that provides a persona to the MTurkAgent.
    """

    def __init__(self, opt, mturk_agent):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_persona_time = opt['max_persona_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        persona_idx, data = self.mturk_agent.persona_generator.pop_persona()
        self.mturk_agent.persona_idx = persona_idx
        self.mturk_agent.persona_data = data
        persona_text = ''
        for s in data:
            persona_text += '<b><span style="color:blue">{}\n</span></b>'.format(s.strip())
        self.mturk_agent.observe({'id': 'SYSTEM', 'show_persona': True, 'text': ONBOARD_MSG + '<br>' + persona_text + '<br>'})
        act = self.mturk_agent.act(timeout=self.max_persona_time)
        if act['episode_done'] or ('text' in act and act['text'] == TIMEOUT_MESSAGE):
            self.mturk_agent.persona_generator.push_persona(self.mturk_agent.persona_idx)
            self.mturk_agent.persona_generator.save_idx_stack()
            self.episodeDone = True
            return
        if 'text' not in act:
            control_msg = {'id': 'SYSTEM', 'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True