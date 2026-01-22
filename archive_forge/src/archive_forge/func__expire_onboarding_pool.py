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
def _expire_onboarding_pool(self):
    """
        Expire any agent that is in an onboarding thread.
        """

    def expire_func(agent):
        self.force_expire_hit(agent.worker_id, agent.assignment_id)

    def is_onboard(agent):
        return agent.get_status() == AssignState.STATUS_ONBOARDING
    self.worker_manager.map_over_agents(expire_func, is_onboard)