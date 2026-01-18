from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
def test_typeSubclass(self) -> None:
    """
        Try to rebuild a base type subclass.
        """
    classDefinition = 'class ListSubclass(list):\n    pass\n'
    exec(classDefinition, self.m.__dict__)
    inst = self.m.ListSubclass()
    inst.append(2)
    exec(classDefinition, self.m.__dict__)
    rebuild.updateInstance(inst)
    self.assertEqual(inst[0], 2)
    self.assertIs(type(inst), self.m.ListSubclass)