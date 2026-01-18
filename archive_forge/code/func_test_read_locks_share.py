from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_read_locks_share(self):
    r_lock = self.read_lock('a-lock-file')
    try:
        lock2 = self.read_lock('a-lock-file')
        lock2.unlock()
    finally:
        r_lock.unlock()