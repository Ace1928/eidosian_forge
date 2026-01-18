from tornado.websocket import WebSocketHandler
import logging
from typing import Dict
Class method used to dispatch the request info to the waiting
        notebook. This method is called in `VoilaHandler` when the request
        info becomes available.
        If this method is called before the opening of websocket connection,
        `msg` is stored in `_cache0` and the message will be dispatched when
        a notebook with corresponding kernel id is connected.

        Args:
            - msg (Dict): this dictionary contains the `kernel_id` to identify
            the waiting notebook and `payload` is the request info.
        