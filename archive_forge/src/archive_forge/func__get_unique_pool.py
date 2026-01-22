import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _get_unique_pool(self, eligibility_function):
    """
        Return a filtered version of the worker pool where each worker is only listed a
        maximum of one time.

        In sandbox this is overridden for testing purposes, and the same worker can be
        returned more than once
        """
    pool = [a for a in self.agent_pool if not a.hit_is_returned]
    if eligibility_function['multiple'] is True:
        agents = eligibility_function['func'](pool)
    else:
        agents = [a for a in pool if eligibility_function['func'](a)]
    unique_agents = []
    unique_worker_ids = []
    for agent in agents:
        if self.is_sandbox or agent.worker_id not in unique_worker_ids:
            unique_agents.append(agent)
            unique_worker_ids.append(agent.worker_id)
    return unique_agents