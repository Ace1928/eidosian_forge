from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_temporary_write_lock(self):
    r_lock = self.read_lock('a-lock-file')
    try:
        status, w_lock = r_lock.temporary_write_lock()
        self.assertTrue(status)
        try:
            self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
        finally:
            r_lock = w_lock.restore_read_lock()
        r_lock2 = self.read_lock('a-lock-file')
        r_lock2.unlock()
    finally:
        r_lock.unlock()