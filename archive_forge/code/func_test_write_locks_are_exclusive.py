from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_write_locks_are_exclusive(self):
    w_lock = self.write_lock('a-lock-file')
    try:
        self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
    finally:
        w_lock.unlock()