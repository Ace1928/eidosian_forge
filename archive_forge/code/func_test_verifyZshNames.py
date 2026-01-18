import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_verifyZshNames(self):
    """
        Using a parameter/flag name that doesn't exist
        will raise an error
        """

    class TmpOptions(FighterAceExtendedOptions):
        compData = Completions(optActions={'detaill': None})
    self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions(), 'ace', BytesIO())

    class TmpOptions2(FighterAceExtendedOptions):
        compData = Completions(mutuallyExclusive=[('foo', 'bar')])
    self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions2(), 'ace', BytesIO())