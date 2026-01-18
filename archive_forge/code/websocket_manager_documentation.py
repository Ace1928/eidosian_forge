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

        This is to restructure a new message to conform to the message structure defined
        in the `chat_service` README.
        