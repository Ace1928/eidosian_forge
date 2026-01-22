import abc
from collections import namedtuple
from datetime import datetime
import pyrfc3339
from ._caveat import parse_caveat
from ._conditions import (
from ._declared import DECLARED_KEY
from ._namespace import Namespace
from ._operation import OP_KEY
from ._time import TIME_KEY
from ._utils import condition_with_prefix
class FirstPartyCaveatChecker(object):
    """Used to check first party caveats for validity with respect to
    information in the provided context.

    If the caveat kind was not recognised, the checker should return
    ErrCaveatNotRecognized.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check_first_party_caveat(self, ctx, caveat):
        """	Checks that the given caveat condition is valid with respect to
        the given context information.
        :param ctx: an Auth context
        :param caveat a string
        """
        raise NotImplementedError('check_first_party_caveat method must be defined in subclass')

    def namespace(self):
        """	Returns the namespace associated with the caveat checker.
        """
        raise NotImplementedError('namespace method must be defined in subclass')