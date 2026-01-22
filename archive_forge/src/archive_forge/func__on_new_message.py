import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def _on_new_message(self, pkt):
    """
        Handle incoming messages from Amazon's SNS queue.

        All other packets should be handled by the worker_manager
        """
    if pkt.sender_id == AMAZON_SNS_NAME:
        self._handle_mturk_message(pkt)
        return
    self.worker_manager.route_packet(pkt)