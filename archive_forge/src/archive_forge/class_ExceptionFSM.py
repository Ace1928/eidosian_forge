import sys
import string
class ExceptionFSM(Exception):
    """This is the FSM Exception class."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'ExceptionFSM: ' + str(self.value)