import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def _get_episode_done_msg(self, text):
    return {'id': self.id, 'text': text, 'episode_done': True}