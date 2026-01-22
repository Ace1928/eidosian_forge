import pkg_resources
import sys
import optparse
from . import bool_optparse
import os
import re
import textwrap
from . import pluginlib
import configparser
import getpass
from logging.config import fileConfig
class BadCommand(Exception):

    def __init__(self, message, exit_code=2):
        self.message = message
        self.exit_code = exit_code
        Exception.__init__(self, message)

    def _get_message(self):
        """Getter for 'message'; needed only to override deprecation
        in BaseException."""
        return self.__message

    def _set_message(self, value):
        """Setter for 'message'; needed only to override deprecation
        in BaseException."""
        self.__message = value
    message = property(_get_message, _set_message)