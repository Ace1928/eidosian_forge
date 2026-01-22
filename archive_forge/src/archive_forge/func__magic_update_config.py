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
def _magic_update_config():
    if wandb.run and wandb.run.config:
        c = wandb.run.config
        user_config = dict(c.items())
        if set(user_config).difference({'magic'}):
            return
    if _magic_get_config('args.absl', None) is False:
        global _args_absl
        _args_absl = None
    if _magic_get_config('args.argparse', None) is False:
        global _args_argparse
        _args_argparse = None
    if _magic_get_config('args.sys', None) is False:
        global _args_system
        _args_system = None
    args = _args_absl or _args_argparse or _args_system
    if args and wandb.run and wandb.run.config:
        wandb.run.config.update(args)