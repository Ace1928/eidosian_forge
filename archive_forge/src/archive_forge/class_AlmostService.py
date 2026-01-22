from zope.interface import implementer
from zope.interface.exceptions import BrokenImplementation
from zope.interface.verify import verifyObject
from twisted.application.service import (
from twisted.persisted.sob import IPersistable
from twisted.trial.unittest import TestCase
@implementer(IService)
class AlmostService:
    """
    Implement IService in a way that can fail.

    In general, classes should maintain invariants that adhere
    to the interfaces that they claim to implement --
    otherwise, it is a bug.

    This is a buggy class -- the IService implementation is fragile,
    and several methods will break it. These bugs are intentional,
    as the tests trigger them -- and then check that the class,
    indeed, no longer complies with the interface (IService)
    that it claims to comply with.

    Since the verification will, by definition, only fail on buggy classes --
    in other words, those which do not actually support the interface they
    claim to support, we have to write a buggy class to properly verify
    the interface.
    """

    def __init__(self, name: str, parent: IServiceCollection, running: bool) -> None:
        self.name = name
        self.parent = parent
        self.running = running

    def makeInvalidByDeletingName(self) -> None:
        """
        Probably not a wise method to call.

        This method removes the :code:`name` attribute,
        which has to exist in IService classes.
        """
        del self.name

    def makeInvalidByDeletingParent(self) -> None:
        """
        Probably not a wise method to call.

        This method removes the :code:`parent` attribute,
        which has to exist in IService classes.
        """
        del self.parent

    def makeInvalidByDeletingRunning(self) -> None:
        """
        Probably not a wise method to call.

        This method removes the :code:`running` attribute,
        which has to exist in IService classes.
        """
        del self.running

    def setName(self, name: object) -> None:
        """
        See L{twisted.application.service.IService}.

        @param name: ignored
        """

    def setServiceParent(self, parent: object) -> None:
        """
        See L{twisted.application.service.IService}.

        @param parent: ignored
        """

    def disownServiceParent(self) -> None:
        """
        See L{twisted.application.service.IService}.
        """

    def privilegedStartService(self) -> None:
        """
        See L{twisted.application.service.IService}.
        """

    def startService(self) -> None:
        """
        See L{twisted.application.service.IService}.
        """

    def stopService(self) -> None:
        """
        See L{twisted.application.service.IService}.
        """