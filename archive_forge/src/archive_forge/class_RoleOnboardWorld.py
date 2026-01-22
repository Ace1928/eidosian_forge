from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.utils.safety import OffensiveStringMatcher
from joblib import Parallel, delayed
from task_config import task_config as config
from extract_and_save_personas import main as main_extract
from constants import (
import numpy as np
import time
import os
import pickle
import random
import copy
from urllib.parse import unquote
class RoleOnboardWorld(MTurkOnboardWorld):
    """
    A world that provides the appropriate instructions during onboarding.
    """

    def __init__(self, opt, mturk_agent, role):
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_onboard_time = opt['max_onboard_time']
        self.role = role
        super().__init__(opt, mturk_agent)

    def parley(self):
        onboard_msg = {'id': 'SYSTEM'}
        onboard_msg['show_persona'] = False
        onboard_msg['text'] = ONBOARD_MSG
        if self.role == WIZARD:
            onboard_msg['role_task_description'] = config['wizard_onboarding']
        else:
            onboard_msg['role_task_description'] = config['apprentice_onboarding']
        self.mturk_agent.observe(onboard_msg)
        act = self.mturk_agent.act(timeout=self.max_onboard_time)
        if act['episode_done'] or ('text' in act and act['text'] == TIMEOUT_MESSAGE):
            self.episodeDone = True
            return
        if 'text' not in act:
            control_msg = {'id': 'SYSTEM', 'text': WAITING_MSG}
            self.mturk_agent.observe(validate(control_msg))
            self.episodeDone = True