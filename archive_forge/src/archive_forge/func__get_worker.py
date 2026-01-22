import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _get_worker(self, worker_id):
    """
        A safe way to get a worker by worker_id.
        """
    return self.mturk_workers.get(worker_id, None)