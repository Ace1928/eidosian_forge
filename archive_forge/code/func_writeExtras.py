import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def writeExtras(self):
    """
        Write out completion information for extra arguments appearing on the
        command-line. These are extra positional arguments not associated
        with a named option. That is, the stuff that gets passed to
        Options.parseArgs().

        @return: L{None}

        @raise ValueError: If C{Completer} with C{repeat=True} is found and
            is not the last item in the C{extraActions} list.
        """
    for i, action in enumerate(self.extraActions):
        if action._repeat and i != len(self.extraActions) - 1:
            raise ValueError('Completer with repeat=True must be last item in Options.extraActions')
        self.file.write(escape(action._shellCode('', usage._ZSH)).encode('utf-8'))
        self.file.write(b' \\\n')