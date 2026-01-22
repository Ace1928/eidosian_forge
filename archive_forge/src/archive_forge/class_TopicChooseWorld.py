from parlai.core.agents import create_agent_from_shared
from parlai.core.message import Message
from parlai.core.worlds import validate, MultiAgentDialogWorld
from parlai.mturk.core.agents import TIMEOUT_MESSAGE
import parlai.mturk.core.mturk_utils as mutils
from parlai.mturk.core.worlds import MTurkOnboardWorld
from parlai.tasks.wizard_of_wikipedia.build import build
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE
from joblib import Parallel, delayed
import json
import numpy as np
import os
import pickle
import random
import time
class TopicChooseWorld(MTurkOnboardWorld):
    """
    A world that provides topics to an MTurk Agent and asks them to choose one.
    """

    def __init__(self, opt, mturk_agent, role='PERSON_1'):
        self.role = role
        self.task_type = 'sandbox' if opt['is_sandbox'] else 'live'
        self.max_choice_time = opt['max_choice_time']
        super().__init__(opt, mturk_agent)

    def parley(self):
        if self.role == 'PERSON_1':
            seen = random.choice([True, False])
            num = random.choice([2, 3])
            topics = self.mturk_agent.topics_generator.get_topics(seen=seen, num=num)
            self.mturk_agent.observe(validate({'id': 'SYSTEM', 'text': PICK_TOPIC_MSG, 'relevant_topics': topics}))
            topic_act = self.mturk_agent.act(timeout=self.max_choice_time)
            timed_out = self.check_timeout(topic_act)
            if timed_out:
                return
            pick_msg = AFTER_PICK_TOPIC_MSG
            self.mturk_agent.observe({'id': 'SYSTEM', 'text': pick_msg})
            self.mturk_agent.chosen_topic = topic_act['text']
            self.mturk_agent.topic_choices = topics
            self.mturk_agent.seen = seen
            self.mturk_agent.observe(validate({'id': 'SYSTEM', 'wait_msg': True}))
        else:
            self.mturk_agent.observe(validate({'id': 'SYSTEM', 'wait_msg': True}))
        return

    def check_timeout(self, act):
        if act['text'] == '[TIMEOUT]' or act['text'] == '[RETURNED]' or act['text'] == '[DISCONNECT]':
            return True
        else:
            return False