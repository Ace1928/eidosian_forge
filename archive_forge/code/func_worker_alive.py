import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def worker_alive(self, worker_id):
    """
        Creates a new worker record if it doesn't exist, returns state.
        """
    if worker_id not in self.mturk_workers:
        self.mturk_workers[worker_id] = WorkerState(worker_id)
    return self.mturk_workers[worker_id]