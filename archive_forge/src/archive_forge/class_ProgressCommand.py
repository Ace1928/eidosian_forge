from __future__ import division
import re
import stat
from .helpers import (
class ProgressCommand(ImportCommand):

    def __init__(self, message):
        ImportCommand.__init__(self, b'progress')
        self.message = message

    def __bytes__(self):
        return b'progress ' + self.message