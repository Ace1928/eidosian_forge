import argparse
import collections
import datetime
import functools
import os
import sys
import time
import uuid
from oslo_utils import encodeutils
import prettytable
from glance.common import exception
import glance.image_cache.client
from glance.version import version_info as version
def user_confirm(prompt, default=False):
    """Yes/No question dialog with user.

    :param prompt: question/statement to present to user (string)
    :param default: boolean value to return if empty string
                    is received as response to prompt

    """
    if default:
        prompt_default = '[Y/n]'
    else:
        prompt_default = '[y/N]'
    answer = input('%s %s ' % (prompt, prompt_default))
    if answer == '':
        return default
    else:
        return answer.lower() in ('yes', 'y')