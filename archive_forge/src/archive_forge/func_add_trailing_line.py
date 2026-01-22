import email.utils
import logging
import os
import re
import socket
from debian.debian_support import Version
def add_trailing_line(self, line):
    """ Add a sign-off (trailer) line to the block """
    self._trailing.append(line)