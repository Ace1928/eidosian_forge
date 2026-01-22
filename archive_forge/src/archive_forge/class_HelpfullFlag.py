from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import errno
import os
import pdb
import sys
import textwrap
import traceback
from absl import command_name
from absl import flags
from absl import logging
class HelpfullFlag(flags.BooleanFlag):
    """Display help for flags in the main module and all dependent modules."""

    def __init__(self):
        super(HelpfullFlag, self).__init__('helpfull', False, 'show full help', allow_hide_cpp=True)

    def parse(self, arg):
        if self._parse(arg):
            usage(writeto_stdout=True)
            sys.exit(1)