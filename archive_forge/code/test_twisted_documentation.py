import sys
from types import ModuleType
from twisted.trial.unittest import TestCase

        Processing of the attributes dictionary is recursive, so a C{dict} value
        it contains may itself contain a C{dict} value to the same effect.
        