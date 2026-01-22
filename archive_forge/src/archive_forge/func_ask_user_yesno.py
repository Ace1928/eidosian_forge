import getpass
import logging
import os
import queue
from cliff.formatters import table
from osc_lib.command import command
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def ask_user_yesno(msg):
    """Ask user Y/N question

    :param str msg: question text
    :return bool: User choice
    """
    while True:
        answer = getpass._raw_input('{} [{}]: '.format(msg, 'y/n'))
        if answer in ('y', 'Y', 'yes'):
            return True
        elif answer in ('n', 'N', 'no'):
            return False