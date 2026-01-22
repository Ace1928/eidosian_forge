from __future__ import annotations
import logging # isort:skip
from typing import Any
from ..message import Empty, Message
class server_info_req(Message[Empty]):
    """ Define the ``SERVER-INFO-REQ`` message for requesting a Bokeh server
    provide information about itself.

    The ``content`` fragment of for this message is empty.

    """
    msgtype = 'SERVER-INFO-REQ'

    @classmethod
    def create(cls, **metadata: Any) -> server_info_req:
        """ Create an ``SERVER-INFO-REQ`` message

        Any keyword arguments will be put into the message ``metadata``
        fragment as-is.

        """
        header = cls.create_header()
        content = Empty()
        return cls(header, metadata, content)