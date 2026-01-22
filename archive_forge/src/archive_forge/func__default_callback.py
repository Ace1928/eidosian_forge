from typing import Dict, TypeVar
from tornado.websocket import WebSocketHandler
import uuid
import logging
import json
def _default_callback(message, socketID):
    logging.warn(f'No callback defined for new WebSocket messages.')