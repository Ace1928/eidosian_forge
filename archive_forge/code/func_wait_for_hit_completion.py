import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def wait_for_hit_completion(self, timeout=None):
    """
        Waits for a hit to be marked as complete.
        """
    WAIT_TIME = 45 * 60
    start_time = time.time()
    while not self.hit_is_complete:
        if time.time() - start_time > WAIT_TIME:
            self.disconnected = True
        if self.hit_is_returned or self.disconnected:
            self.m_free_workers([self])
            return False
        time.sleep(shared_utils.THREAD_MEDIUM_SLEEP)
    shared_utils.print_and_log(logging.INFO, 'Conversation ID: {}, Agent ID: {} - HIT is done.'.format(self.conversation_id, self.id))
    self.m_free_workers([self])
    return True