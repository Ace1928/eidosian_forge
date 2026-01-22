import os
from collections import namedtuple
import re
import sqlite3
import typing
import warnings
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects.packages_utils import (get_packagepath,
from collections import OrderedDict
class HelpNotFoundError(KeyError):
    """ Exception raised when an help topic cannot be found. """

    def __init__(self, msg, topic=None, package=None):
        super(HelpNotFoundError, self).__init__(msg)
        self.topic = topic
        self.package = package