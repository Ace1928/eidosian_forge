import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class OriginalLocationTests(_WritesPythonModules):
    """
    Tests that L{isOriginalLocation} detects when a
    L{PythonAttribute}'s FQPN refers to an object inside the module
    where it was defined.

    For example: A L{twisted.python.modules.PythonAttribute} with a
    name of 'foo.bar' that refers to a 'bar' object defined in module
    'baz' does *not* refer to bar's original location, while a
    L{PythonAttribute} with a name of 'baz.bar' does.

    """

    def setUp(self):
        super(OriginalLocationTests, self).setUp()
        from .._discover import isOriginalLocation
        self.isOriginalLocation = isOriginalLocation

    def test_failsWithNoModule(self):
        """
        L{isOriginalLocation} returns False when the attribute refers to an
        object whose source module cannot be determined.
        """
        source = '        class Fake(object):\n            pass\n        hasEmptyModule = Fake()\n        hasEmptyModule.__module__ = None\n        '
        moduleDict = self.makeModuleAsDict(source, self.pathDir, 'empty_module_attr.py')
        self.assertFalse(self.isOriginalLocation(moduleDict['empty_module_attr.hasEmptyModule']))

    def test_failsWithDifferentModule(self):
        """
        L{isOriginalLocation} returns False when the attribute refers to
        an object outside of the module where that object was defined.
        """
        originalSource = '        class ImportThisClass(object):\n            pass\n        importThisObject = ImportThisClass()\n        importThisNestingObject = ImportThisClass()\n        importThisNestingObject.nestedObject = ImportThisClass()\n        '
        importingSource = '        from original import (ImportThisClass,\n                              importThisObject,\n                              importThisNestingObject)\n        '
        self.makeModule(originalSource, self.pathDir, 'original.py')
        importingDict = self.makeModuleAsDict(importingSource, self.pathDir, 'importing.py')
        self.assertFalse(self.isOriginalLocation(importingDict['importing.ImportThisClass']))
        self.assertFalse(self.isOriginalLocation(importingDict['importing.importThisObject']))
        nestingObject = importingDict['importing.importThisNestingObject']
        nestingObjectDict = self.attributesAsDict(nestingObject)
        nestedObject = nestingObjectDict['importing.importThisNestingObject.nestedObject']
        self.assertFalse(self.isOriginalLocation(nestedObject))

    def test_succeedsWithSameModule(self):
        """
        L{isOriginalLocation} returns True when the attribute refers to an
        object inside the module where that object was defined.
        """
        mSource = textwrap.dedent('\n        class ThisClassWasDefinedHere(object):\n            pass\n        anObject = ThisClassWasDefinedHere()\n        aNestingObject = ThisClassWasDefinedHere()\n        aNestingObject.nestedObject = ThisClassWasDefinedHere()\n        ')
        mDict = self.makeModuleAsDict(mSource, self.pathDir, 'm.py')
        self.assertTrue(self.isOriginalLocation(mDict['m.ThisClassWasDefinedHere']))
        self.assertTrue(self.isOriginalLocation(mDict['m.aNestingObject']))
        nestingObject = mDict['m.aNestingObject']
        nestingObjectDict = self.attributesAsDict(nestingObject)
        nestedObject = nestingObjectDict['m.aNestingObject.nestedObject']
        self.assertTrue(self.isOriginalLocation(nestedObject))