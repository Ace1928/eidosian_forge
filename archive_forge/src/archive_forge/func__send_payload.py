import time
from abc import ABC, abstractmethod
from queue import Queue
from parlai.core.agents import Agent
def _send_payload(self, receiver_id, data, quick_replies=None, persona_id=None):
    """
        Send a payload through the message manager.

        :param receiver_id:
            int identifier for agent to send message to
        :param data:
            object data to send
        :param quick_replies:
            list of quick replies
        :param persona_id:
            identifier of persona
        :return:
            a dictionary of a json response from the manager observing a payload
        """
    return self.manager.observe_payload(receiver_id, data, quick_replies, persona_id)