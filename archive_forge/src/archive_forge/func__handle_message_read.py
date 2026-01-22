import json
import asyncio
import logging
from parlai.core.agents import create_agent
from parlai.chat_service.core.chat_service_manager import ChatServiceManager
import parlai.chat_service.utils.logging as log_utils
import parlai.chat_service.utils.misc as utils
from parlai.chat_service.services.websocket.sockets import MessageSocketHandler
from agents import WebsocketAgent
import tornado
from tornado.options import options
def _handle_message_read(self, event):
    """
        Send read receipt back to user who sent message This function is left empty as
        it is not applicable to websockets since read receipts are not supported.
        """
    pass