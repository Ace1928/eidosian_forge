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
class ChatServiceMessageSender(ABC):
    """
        ChatServiceMessageSender is a wrapper around requests that simplifies the the
        process of sending content.
        """

    @abstractmethod
    def send_read(self, receiver_id: int):
        """
            Send read receipt to agent at receiver_id.
            """
        pass

    @abstractmethod
    def typing_on(self, receiver_id: int, persona_id: str=None):
        """
            Send typing on msg to agent at receiver_id.
            """
        pass