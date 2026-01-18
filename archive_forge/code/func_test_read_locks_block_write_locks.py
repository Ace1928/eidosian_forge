from .. import debug, errors, lock, tests
from .scenarios import load_tests_apply_scenarios
def test_read_locks_block_write_locks(self):
    r_lock = self.read_lock('a-lock-file')
    try:
        if lock.have_fcntl and self.write_lock is lock._fcntl_WriteLock:
            debug.debug_flags.add('strict_locks')
            self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
            debug.debug_flags.remove('strict_locks')
            try:
                w_lock = self.write_lock('a-lock-file')
            except errors.LockContention:
                self.fail('Unexpected success. fcntl read locks do not usually block write locks')
            else:
                w_lock.unlock()
                self.knownFailure("fcntl read locks don't block write locks without -Dlock")
        else:
            self.assertRaises(errors.LockContention, self.write_lock, 'a-lock-file')
    finally:
        r_lock.unlock()