import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime

        @param mainMod: The '__main__' module that this class will proxy.
        