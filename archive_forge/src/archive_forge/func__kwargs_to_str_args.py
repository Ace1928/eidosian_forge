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
def _kwargs_to_str_args(self, **kwargs):
    """
        Attempt to map from python-code kwargs into CLI args.

        e.g. model_file -> --model-file.

        Works with short options too, like t="convai2".
        """
    kwname_to_action = {}
    for action in self._actions:
        if action.dest == 'help':
            continue
        for option_string in action.option_strings:
            kwname = option_string.lstrip('-').replace('-', '_')
            assert kwname not in kwname_to_action or kwname_to_action[kwname] is action, f'No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})'
            kwname_to_action[kwname] = action
    string_args = []
    for kwname, value in kwargs.items():
        if kwname not in kwname_to_action:
            continue
        action = kwname_to_action[kwname]
        last_option_string = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                string_args.append(last_option_string)
        elif isinstance(action, argparse._StoreAction) and action.nargs is None:
            string_args.append(last_option_string)
            string_args.append(self._value2argstr(value))
        elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
            string_args.append(last_option_string)
            string_args.extend([self._value2argstr(value) for v in value])
        else:
            raise TypeError(f"Don't know what to do with {action}")
    self.add_extra_args(string_args)
    kwname_to_action = {}
    for action in self._actions:
        if action.dest == 'help':
            continue
        for option_string in action.option_strings:
            kwname = option_string.lstrip('-').replace('-', '_')
            assert kwname not in kwname_to_action or kwname_to_action[kwname] is action, f'No duplicate names! ({kwname}, {kwname_to_action[kwname]}, {action})'
            kwname_to_action[kwname] = action
    string_args = []
    for kwname, value in kwargs.items():
        action = kwname_to_action[kwname]
        last_option_string = action.option_strings[-1]
        if isinstance(action, argparse._StoreTrueAction):
            if bool(value):
                string_args.append(last_option_string)
        elif isinstance(action, argparse._StoreAction) and action.nargs is None:
            string_args.append(last_option_string)
            string_args.append(self._value2argstr(value))
        elif isinstance(action, argparse._StoreAction) and action.nargs in '*+':
            string_args.append(last_option_string)
            string_args.extend([str(v) for v in value])
        else:
            raise TypeError(f"Don't know what to do with {action}")
    return string_args