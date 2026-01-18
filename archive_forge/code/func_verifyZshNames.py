import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType
def verifyZshNames(self):
    """
        Ensure that none of the option names given in the metadata are typoed
        @return: L{None}
        @raise ValueError: If unknown option names have been found.
        """

    def err(name):
        raise ValueError('Unknown option name "%s" found while\nexamining Completions instances on %s' % (name, self.options))
    for name in itertools.chain(self.descriptions, self.optActions, self.multiUse):
        if name not in self.allOptionsNameToDefinition:
            err(name)
    for seq in self.mutuallyExclusive:
        for name in seq:
            if name not in self.allOptionsNameToDefinition:
                err(name)