import argparse
import copy
import json
import os
import re
import sys
import yaml
import wandb
from wandb import trigger
from wandb.util import add_import_hook, get_optional_module
def _process_system_args():
    global _args_system
    parser = SafeArgumentParser(add_help=False)
    for num, arg in enumerate(sys.argv):
        try:
            next_arg = sys.argv[num + 1]
        except IndexError:
            next_arg = ''
        if arg.startswith(('-', '--')) and (not next_arg.startswith(('-', '--'))):
            try:
                parser.add_argument(arg)
            except ValueError:
                pass
    try:
        parsed, unknown = parser.parse_known_args()
    except ArgumentException:
        pass
    else:
        _args_system = vars(parsed)