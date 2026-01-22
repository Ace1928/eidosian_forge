import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
class AgentTimeoutError(AbsentAgentError):
    """
    Exception for when a worker doesn't respond in time.
    """

    def __init__(self, timeout, worker_id, assignment_id):
        super().__init__(f'Agent exceeded {timeout}', worker_id, assignment_id)