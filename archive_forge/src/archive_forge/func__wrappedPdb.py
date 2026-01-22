import gc
import inspect
import os
import pdb
import random
import sys
import time
import trace
import warnings
from typing import NoReturn, Optional, Type
from twisted import plugin
from twisted.application import app
from twisted.internet import defer
from twisted.python import failure, reflect, usage
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedModule
from twisted.trial import itrial, runner
from twisted.trial._dist.disttrial import DistTrialRunner
from twisted.trial.unittest import TestSuite
def _wrappedPdb():
    """
    Wrap an instance of C{pdb.Pdb} with readline support and load any .rcs.

    """
    dbg = pdb.Pdb()
    try:
        namedModule('readline')
    except ImportError:
        print('readline module not available')
    for path in ('.pdbrc', 'pdbrc'):
        if os.path.exists(path):
            try:
                rcFile = open(path)
            except OSError:
                pass
            else:
                with rcFile:
                    dbg.rcLines.extend(rcFile.readlines())
    return dbg