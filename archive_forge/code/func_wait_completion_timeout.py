import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def wait_completion_timeout(self, iterations):
    """
        Suspends the thread waiting for hit completion for some number of iterations on
        the THREAD_MTURK_POLLING_SLEEP time.
        """
    iters = shared_utils.THREAD_MTURK_POLLING_SLEEP / shared_utils.THREAD_MEDIUM_SLEEP
    i = 0
    while not self.hit_is_complete and i < iters * iterations:
        time.sleep(shared_utils.THREAD_SHORT_SLEEP)
        i += 1
    return