from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import itertools
import os
import pkgutil
import re
import six
from subprocess import PIPE
from subprocess import Popen
import gslib.addlhelp
from gslib.command import Command
from gslib.command import OLD_ALIAS_MAP
import gslib.commands
from gslib.exception import CommandException
from gslib.help_provider import HelpProvider
from gslib.help_provider import MAX_HELP_NAME_LEN
from gslib.utils import constants
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.system_util import IsRunningInteractively
from gslib.utils.system_util import GetTermLines
from gslib.utils import text_util
def _OutputHelp(self, help_str):
    """Outputs simply formatted string.

    This function paginates if the string is too long, PAGER is defined, and
    the output is a tty.

    Args:
      help_str: String to format.
    """
    if IS_WINDOWS or not IsRunningInteractively():
        help_str = re.sub('<B>', '', help_str)
        help_str = re.sub('</B>', '', help_str)
        text_util.print_to_fd(help_str)
        return
    help_str = re.sub('<B>', '\x1b[1m', help_str)
    help_str = re.sub('</B>', '\x1b[0;0m', help_str)
    num_lines = len(help_str.split('\n'))
    if 'PAGER' in os.environ and num_lines >= GetTermLines():
        pager = os.environ['PAGER'].split(' ')
        if pager[0].endswith('less'):
            pager.append('-r')
        try:
            if six.PY2:
                input_for_pager = help_str.encode(constants.UTF8)
            else:
                input_for_pager = help_str
            Popen(pager, stdin=PIPE, universal_newlines=True).communicate(input=input_for_pager)
        except OSError as e:
            raise CommandException('Unable to open pager (%s): %s' % (' '.join(pager), e))
    else:
        text_util.print_to_fd(help_str)