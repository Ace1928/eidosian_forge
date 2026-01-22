import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.shared_utils as shared_utils
def _print_not_available_for(self, item):
    shared_utils.print_and_log(logging.WARN, 'Conversation ID: {}, Agent ID: {} - HIT is abandoned and thus not available for {}.'.format(self.conversation_id, self.id, item), should_print=True)