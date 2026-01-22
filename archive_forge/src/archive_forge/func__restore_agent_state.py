import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.socket_manager import Packet, SocketManager, StaticSocketManager
from parlai.mturk.core.worker_manager import WorkerManager
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.mturk_utils as mturk_utils
import parlai.mturk.core.server_utils as server_utils
import parlai.mturk.core.shared_utils as shared_utils
def _restore_agent_state(self, worker_id, assignment_id):
    """
        Send a command to restore the state of an agent who reconnected.
        """
    agent = self.worker_manager._get_agent(worker_id, assignment_id)
    if agent is not None:
        agent.alived = False

        def send_state_data():
            while not agent.alived and (not agent.hit_is_expired):
                time.sleep(shared_utils.THREAD_SHORT_SLEEP)
            data = {'text': data_model.COMMAND_RESTORE_STATE, 'messages': agent.get_messages(), 'last_command': agent.get_last_command()}
            self.send_command(worker_id, assignment_id, data)
            if agent.message_request_time is not None:
                agent.request_message()
        state_thread = threading.Thread(name='restore-agent-{}'.format(agent.worker_id), target=send_state_data)
        state_thread.daemon = True
        state_thread.start()
        self.worker_manager.change_agent_conversation(agent=agent, conversation_id=agent.conversation_id, new_agent_id=agent.id)