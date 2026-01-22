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
def _launch_overworld(self, agent_id: int):
    """
        Launch an overworld for the given agent id, replacing the existing overworld if
        it exists already.
        """
    agent_state = self.get_agent_state(agent_id)
    task_id = 'overworld-{}-{}'.format(agent_id, time.time())
    if agent_state is None:
        agent = self._create_agent(task_id, agent_id)
        agent_state = AgentState(agent_id, agent)
        self.messenger_agent_states[agent_id] = agent_state
    else:
        agent = agent_state.overworld_agent
        self.agent_id_to_overworld_future[agent_id].cancel()
    future = self.world_runner.launch_overworld(task_id, self.overworld, self.onboard_map, agent)

    def _done_callback(fut):
        e = fut.exception()
        if e is not None:
            log_utils.print_and_log(logging.ERROR, 'World {} had error {}'.format(task_id, repr(e)), should_print=True)
            traceback.print_exc(file=sys.stdout)
            if self.debug:
                raise e
    future.add_done_callback(_done_callback)
    self.agent_id_to_overworld_future[agent_id] = future