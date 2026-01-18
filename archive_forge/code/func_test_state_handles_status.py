import unittest
import os
import time
from unittest import mock
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model
def test_state_handles_status(self):
    """
        Ensures status updates and is_final are valid.
        """
    for status in statuses:
        self.agent_state1.set_status(status)
        self.assertEqual(self.agent_state1.get_status(), status)
    for status in active_statuses:
        self.agent_state1.set_status(status)
        self.assertFalse(self.agent_state1.is_final())
    for status in complete_statuses:
        self.agent_state1.set_status(status)
        self.assertTrue(self.agent_state1.is_final())