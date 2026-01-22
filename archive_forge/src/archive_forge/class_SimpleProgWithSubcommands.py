import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class SimpleProgWithSubcommands(SimpleProgOptions):
    optFlags = [['some-option'], ['other-option', 'o']]
    optParameters = [['some-param'], ['other-param', 'p'], ['another-param', 'P', 'Yet Another Param']]
    subCommands = [['sub1', None, SimpleProgSub1, 'Sub Command 1'], ['sub2', None, SimpleProgSub2, 'Sub Command 2']]