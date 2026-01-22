from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
class CrefUtilTests(TestCase):
    """
    Tests for L{crefutil}.
    """

    def test_dictUnknownKey(self) -> None:
        """
        L{crefutil._DictKeyAndValue} only support keys C{0} and C{1}.
        """
        d = crefutil._DictKeyAndValue({})
        self.assertRaises(RuntimeError, d.__setitem__, 2, 3)

    def test_deferSetMultipleTimes(self) -> None:
        """
        L{crefutil._Defer} can be assigned a key only one time.
        """
        d = crefutil._Defer()
        d[0] = 1
        self.assertRaises(RuntimeError, d.__setitem__, 0, 1)

    def test_containerWhereAllElementsAreKnown(self) -> None:
        """
        A L{crefutil._Container} where all of its elements are known at
        construction time is nonsensical and will result in errors in any call
        to addDependant.
        """
        container = crefutil._Container([1, 2, 3], list)
        self.assertRaises(AssertionError, container.addDependant, {}, 'ignore-me')

    def test_dontPutCircularReferencesInDictionaryKeys(self) -> None:
        """
        If a dictionary key contains a circular reference (which is probably a
        bad practice anyway) it will be resolved by a
        L{crefutil._DictKeyAndValue}, not by placing a L{crefutil.NotKnown}
        into a dictionary key.
        """
        self.assertRaises(AssertionError, dict().__setitem__, crefutil.NotKnown(), 'value')

    def test_dontCallInstanceMethodsThatArentReady(self) -> None:
        """
        L{crefutil._InstanceMethod} raises L{AssertionError} to indicate it
        should not be called.  This should not be possible with any of its API
        clients, but is provided for helping to debug.
        """
        self.assertRaises(AssertionError, crefutil._InstanceMethod('no_name', crefutil.NotKnown(), type))