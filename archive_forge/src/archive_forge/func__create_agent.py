import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _create_agent(self, hit_id, assignment_id, worker_id):
    """
        Initialize an agent and return it.
        """
    return MTurkAgent(self.opt, self.mturk_manager, hit_id, assignment_id, worker_id)