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
def test_nullVersionUpgrade(self) -> None:
    global NullVersioned

    class NullVersioned:

        def __init__(self) -> None:
            self.ok = 0
    pkcl = pickle.dumps(NullVersioned())

    class NullVersioned(styles.Versioned):
        persistenceVersion = 1

        def upgradeToVersion1(self) -> None:
            self.ok = 1
    mnv = pickle.loads(pkcl)
    styles.doUpgrade()
    assert mnv.ok, 'initial upgrade not run!'