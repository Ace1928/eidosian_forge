from twisted.trial.unittest import TestCase
from twisted.internet.defer import Deferred, fail, succeed
from .._resultstore import ResultStore
from .._eventloop import EventualResult

        Unretrieved EventualResults have their errors, if any, logged on
        shutdown.
        