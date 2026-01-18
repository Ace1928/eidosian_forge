import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
def test_unknownShell(self):
    """
        Using an unknown shellType should raise NotImplementedError
        """
    action = _shellcomp.SubcommandAction()
    self.assertRaises(NotImplementedError, action._shellCode, None, 'bad_shell_type')