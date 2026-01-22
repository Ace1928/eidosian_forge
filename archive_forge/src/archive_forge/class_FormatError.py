import sys
import re
import os
from configparser import RawConfigParser
class FormatError(OSError):
    """
    Exception thrown when there is a problem parsing a configuration file.

    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg