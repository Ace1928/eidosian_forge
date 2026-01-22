from pyparsing import *
from sys import stdin, argv, exit
class ExceptionSharedData(object):
    """Class for exception handling data"""

    def __init__(self):
        self.location = 0
        self.text = ''

    def setpos(self, location, text):
        """Helper function for setting curently parsed text and position"""
        self.location = location
        self.text = text