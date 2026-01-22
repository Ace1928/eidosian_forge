import abc
from collections import namedtuple
from ._error import (
from ._codec import decode_caveat
from ._macaroon import (
from ._versions import VERSION_2
from ._third_party import ThirdPartyCaveatInfo
import macaroonbakery.checkers as checkers
class ThirdPartyCaveatChecker(object):
    """ Defines an abstract class that's used to check third party caveats.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check_third_party_caveat(self, ctx, info):
        """ If the caveat is valid, it returns optionally a slice of
        extra caveats that will be added to the discharge macaroon.
        If the caveat kind was not recognised, the checker should
        raise a CaveatNotRecognized exception; if the check failed,
        it should raise a ThirdPartyCaveatCheckFailed exception.
        :param ctx (AuthContext)
        :param info (ThirdPartyCaveatInfo) holds the information decoded from
        a third party caveat id
        :return: An array of extra caveats to be added to the discharge
        macaroon.
        """
        raise NotImplementedError('check_third_party_caveat method must be defined in subclass')