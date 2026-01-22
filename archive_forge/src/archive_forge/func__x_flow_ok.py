import logging
import socket
from collections import defaultdict
from queue import Queue
from vine import ensure_promise
from . import spec
from .abstract_channel import AbstractChannel
from .exceptions import (ChannelError, ConsumerCancelled, MessageNacked,
from .protocol import queue_declare_ok_t
def _x_flow_ok(self, active):
    """Confirm a flow method.

        Confirms to the peer that a flow command was received and
        processed.

        PARAMETERS:
            active: boolean

                current flow setting

                Confirms the setting of the processed flow method:
                True means the peer will start sending or continue
                to send content frames; False means it will not.
        """
    return self.send_method(spec.Channel.FlowOk, 'b', (active,))