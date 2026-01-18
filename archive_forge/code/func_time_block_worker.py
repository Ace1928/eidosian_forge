import logging
import os
import pickle
import time
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def time_block_worker(self, worker_id):
    self.time_blocked_workers.append(worker_id)
    self.mturk_manager.soft_block_worker(worker_id, 'max_time_qual')