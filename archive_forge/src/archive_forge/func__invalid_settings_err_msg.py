import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
def _invalid_settings_err_msg(settings, verbose=False):
    valid_settings = ', '.join(['all'] + list(log_registry.log_alias_to_log_qnames.keys()) + list(log_registry.artifact_names))
    msg = f'\nInvalid log settings: {settings}, must be a comma separated list of fully\nqualified module names, registered log names or registered artifact names.\nFor more info on various settings, try TORCH_LOGS="help"\nValid settings:\n{valid_settings}\n'
    return msg