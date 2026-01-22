import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def add_agent(self, mturk_agent):
    """
        Add an assignment to this worker state with the given assign_it.
        """
    assert mturk_agent.worker_id == self.worker_id, "Can't add agent that does not match state's worker_id"
    self.agents[mturk_agent.assignment_id] = mturk_agent