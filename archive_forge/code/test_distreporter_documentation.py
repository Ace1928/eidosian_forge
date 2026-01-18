from io import StringIO
from twisted.python.failure import Failure
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial.reporter import TreeReporter
from twisted.trial.unittest import TestCase

        Calling methods of L{DistReporter} add calls to the running queue of
        the test.
        