from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, errors
class WrapperAddressProtocol(ABC):
    """Base class for Wrapper Address Protocol.

    Objects that inherit from the WrapperAddressProtocol can be passed
    as arguments to Numba jit compiled functions where it can be used
    as first-class functions. As a minimum, the derived types must
    implement two methods ``__wrapper_address__`` and ``signature``.
    """

    @abstractmethod
    def __wrapper_address__(self):
        """Return the address of a first-class function.

        Returns
        -------
        addr : int
        """

    @abstractmethod
    def signature(self):
        """Return the signature of a first-class function.

        Returns
        -------
        sig : Signature
          The returned Signature instance represents the type of a
          first-class function that the given WrapperAddressProtocol
          instance represents.
        """