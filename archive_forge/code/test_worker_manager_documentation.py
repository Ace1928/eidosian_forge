import unittest
import os
from unittest import mock
from parlai.mturk.core.dev.worker_manager import WorkerManager, WorkerState
from parlai.mturk.core.dev.agents import MTurkAgent, AssignState
from parlai.mturk.core.dev.mturk_manager import MTurkManager
from parlai.core.params import ParlaiParser
import parlai.mturk.core.dev.worker_manager as WorkerManagerFile
import parlai.mturk.core.dev.data_model as data_model

        Tests handling conversation state, moving agents to the correct conversations,
        and disconnecting one worker in an active convo.
        