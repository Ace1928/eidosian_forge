import logging
import os
import threading
import time
import uuid
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet
from parlai.mturk.webapp.run_mocks.mock_turk_agent import MockTurkAgent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def timeout_all_agents(self):
    """
        Set all agent statuses to disconnect to kill the world.
        """
    for agent in self.agents:
        agent.disconnected = True