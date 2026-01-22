import time
from abc import ABC, abstractmethod
from queue import Queue
from parlai.core.agents import Agent
def act_blocking(self, timeout=None):
    """
        Repeatedly loop until we retrieve a message from the queue.
        """
    while True:
        if self.message_request_time is None:
            self.message_request_time = time.time()
        msg = self.act()
        if msg is not None:
            self.message_request_time = None
            return msg
        if self._check_timeout(timeout):
            return None
        time.sleep(0.2)