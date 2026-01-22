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
def _add_agent_to_pool(self, agent):
    """
        Add a single agent to the pool.
        """
    if agent not in self.agent_pool:
        with self.agent_pool_change_condition:
            if agent not in self.agent_pool:
                shared_utils.print_and_log(logging.DEBUG, 'Adding worker {} to pool.'.format(agent.worker_id))
                self.agent_pool.append(agent)