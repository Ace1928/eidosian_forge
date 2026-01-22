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
def _expire_all_conversations(self):
    """
        Iterate through all sub-worlds and shut them down.
        """
    self.running = False
    for agent_id, overworld_fut in self.agent_id_to_overworld_future.items():
        self.observe_message(agent_id, 'System: The conversation bot going to go offline soon, finish any active conversations in the next 5 minutes.')
        overworld_fut.cancel()
    time_passed = 0
    while time_passed < 5 * 60:
        any_alive = False
        for task_fut in self.active_worlds.values():
            if task_fut is not None:
                any_alive = any_alive or task_fut.running()
        if not any_alive:
            break
        time.sleep(1)
    self.shutting_down = True