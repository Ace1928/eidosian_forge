from breezy import errors, osutils
from breezy.tests import features
from breezy.tests.per_lock import TestCaseWithLock
We can only grab a write lock if all read locks are done.