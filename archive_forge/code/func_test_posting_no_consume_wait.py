import contextlib
import threading
import time
from taskflow import exceptions as excp
from taskflow.persistence.backends import impl_dir
from taskflow import states
from taskflow.tests import utils as test_utils
from taskflow.utils import persistence_utils as p_utils
from taskflow.utils import threading_utils
def test_posting_no_consume_wait(self):
    with connect_close(self.board):
        jb = self.board.post('test', p_utils.temporary_log_book())
        self.assertFalse(jb.wait(0.1))