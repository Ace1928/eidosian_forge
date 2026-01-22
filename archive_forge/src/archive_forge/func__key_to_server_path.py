import ast
import sys
from typing import Any
from .internal import Filters, Key
def _key_to_server_path(key: Key):
    name = key.name
    section = key.section
    if section == 'config':
        return f'config.{name}'
    elif section == 'summary':
        return f'summary_metrics.{name}'
    elif section == 'keys_info':
        return f'keys_info.keys.{name}'
    elif section == 'tags':
        return f'tags.{name}'
    elif section == 'runs':
        return name
    raise ValueError(f'Invalid key ({key})')