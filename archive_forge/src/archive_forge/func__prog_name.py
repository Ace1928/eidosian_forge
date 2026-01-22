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
def _prog_name(self):
    return '%s %s' % (os.path.basename(sys.argv[0]), self.command_name)