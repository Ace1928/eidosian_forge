from random import random as _goodEnoughRandom
from typing import List
from automat import MethodicalMachine
from twisted.application import service
from twisted.internet import task
from twisted.internet.defer import (
from twisted.logger import Logger
from twisted.python import log
from twisted.python.failure import Failure
def _firstResult(gen):
    """
    Return the first element of a generator and exhaust it.

    C{MethodicalMachine.upon}'s C{collector} argument takes a generator of
    output results. If the generator is exhausted, the later outputs aren't
    actually run.

    @param gen: Generator to extract values from

    @return: The first element of the generator.
    """
    return list(gen)[0]