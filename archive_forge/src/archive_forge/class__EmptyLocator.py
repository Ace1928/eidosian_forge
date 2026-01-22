import abc
from collections import namedtuple
from ._error import (
from ._codec import decode_caveat
from ._macaroon import (
from ._versions import VERSION_2
from ._third_party import ThirdPartyCaveatInfo
import macaroonbakery.checkers as checkers
class _EmptyLocator(ThirdPartyLocator):

    def third_party_info(self, loc):
        return None