import getpass
import logging
import os
import queue
from cliff.formatters import table
from osc_lib.command import command
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
Ask user Y/N question

    :param str msg: question text
    :return bool: User choice
    