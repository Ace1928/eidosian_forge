from abc import ABC, abstractmethod
from asyncio import Future
import copy
import sys
import logging
import datetime
import threading
import time
import traceback
from typing import Dict, Any, Optional, List, Callable
from parlai.chat_service.core.agents import ChatServiceAgent
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
import parlai.chat_service.utils.server as server_utils
from parlai.chat_service.core.world_runner import ChatServiceWorldRunner
from parlai.core.opt import Opt
def _remove_agent(self, agent_id: int):
    """
        Remove an agent from the system (after they disconnect or leave in some other
        way)
        """
    self.observe_message(agent_id, 'See you later!')
    for world_type in self.agent_pool:
        agent_state = self.get_agent_state(agent_id)
        if agent_state in self.agent_pool[world_type]:
            assert agent_state is not None
            self.agent_pool[world_type].remove(agent_state)
            self.remove_agent_from_pool(agent_state, world_type=world_type)
    del self.messenger_agent_states[agent_id]
    del self.agent_id_to_overworld_future[agent_id]