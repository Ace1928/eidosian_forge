import os
import sys
import uuid
import traceback
import json
import param
from ._version import __version__
@classmethod
def _process_comm_msg(cls, msg):
    """
        Processes comm messages to handle global actions such as
        cleaning up plots.
        """
    event_type = msg['event_type']
    if event_type == 'delete':
        for action in cls._delete_actions:
            action(msg['id'])
    elif event_type == 'server_delete':
        for action in cls._server_delete_actions:
            action(msg['id'])