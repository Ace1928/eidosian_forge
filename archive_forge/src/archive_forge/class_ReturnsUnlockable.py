from testtools.matchers import Equals, Matcher, Mismatch
from .. import osutils
from .. import revision as _mod_revision
from ..tree import InterTree, TreeChange
class ReturnsUnlockable(Matcher):
    """A matcher that checks for the pattern we want lock* methods to have:

    They should return an object with an unlock() method.
    Calling that method should unlock the original object.

    :ivar lockable_thing: The object which can be locked that will be
        inspected.
    """

    def __init__(self, lockable_thing):
        Matcher.__init__(self)
        self.lockable_thing = lockable_thing

    def __str__(self):
        return 'ReturnsUnlockable(lockable_thing=%s)' % self.lockable_thing

    def match(self, lock_method):
        lock_method().unlock()
        if self.lockable_thing.is_locked():
            return _IsLocked(self.lockable_thing)
        return None