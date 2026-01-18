import types
from twisted.internet.defer import Deferred, ensureDeferred, fail, succeed
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase

        Yielding from a paused & chained Deferred will give the result when it
        has one.
        