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
def _handle_custom_options(self, kwargs):
    """
        Handle custom parlai options.

        Includes hidden, recommended. Future may include no_save and no_override.
        """
    action_attr = {}
    if 'recommended' in kwargs:
        rec = kwargs.pop('recommended')
        action_attr['recommended'] = rec
    action_attr['hidden'] = kwargs.get('hidden', False)
    action_attr['real_help'] = kwargs.get('help', None)
    if 'hidden' in kwargs:
        if kwargs.pop('hidden'):
            kwargs['help'] = argparse.SUPPRESS
    if 'type' in kwargs and kwargs['type'] is bool:
        kwargs['type'] = 'bool'
    return (kwargs, action_attr)