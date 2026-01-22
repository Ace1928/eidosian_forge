import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest
class ZshScriptTestMixin(metaclass=ZshScriptTestMeta):
    """
    Integration test helper to show that C{usage.Options} classes can have zsh
    completion functions generated for them without raising errors.

    In your subclasses set a class variable like so::

      #            | cmd name | Fully Qualified Python Name of Options class |
      #
      generateFor = [('conch',  'twisted.conch.scripts.conch.ClientOptions'),
                     ('twistd', 'twisted.scripts.twistd.ServerOptions'),
                     ]

    Each package that contains Twisted scripts should contain one TestCase
    subclass which also inherits from this mixin, and contains a C{generateFor}
    list appropriate for the scripts in that package.
    """