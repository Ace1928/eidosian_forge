import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_posting_claim_diff_owner(self):
    with connect_close(self.board):
        with self.flush(self.client):
            self.board.post('test', p_utils.temporary_log_book())
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(1, len(possible_jobs))
        with self.flush(self.client):
            self.board.claim(possible_jobs[0], self.board.name)
        possible_jobs = list(self.board.iterjobs())
        self.assertEqual(1, len(possible_jobs))
        self.assertRaises(excp.UnclaimableJob, self.board.claim, possible_jobs[0], self.board.name + '-1')
        possible_jobs = list(self.board.iterjobs(only_unclaimed=True))
        self.assertEqual(0, len(possible_jobs))