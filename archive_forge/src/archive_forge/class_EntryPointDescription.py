import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
class EntryPointDescription(object):

    def __init__(self, group):
        self.group = group