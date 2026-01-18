import unittest
import os
import time
import threading
from unittest import mock
from parlai.mturk.core.agents import MTurkAgent
from parlai.mturk.core.shared_utils import AssignState
from parlai.mturk.core.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.worker_manager as WorkerManagerFile
import parlai.mturk.core.data_model as data_model
def wait_for_status_wrap():
    nonlocal has_changed
    self.turk_agent.wait_for_status(AssignState.STATUS_WAITING)
    has_changed = True