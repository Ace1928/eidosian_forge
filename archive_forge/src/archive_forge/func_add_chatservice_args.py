import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_chatservice_args(self):
    """
        Arguments for all chat services.
        """
    args = self.add_argument_group('Chat Services')
    args.add_argument('--debug', dest='is_debug', action='store_true', help='print and log all server interactions and messages')
    args.add_argument('--config-path', default=None, type=str, help='/path/to/config/file for a given task.')
    args.add_argument('--password', dest='password', type=str, default=None, help='Require a password for entry to the bot')