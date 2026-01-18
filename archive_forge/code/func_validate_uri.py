import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import yaml
def validate_uri(uri: str):
    if not isinstance(uri, str):
        raise TypeError(f'URIs for working_dir and py_modules must be strings, got {type(uri)}.')
    try:
        from ray._private.runtime_env.packaging import parse_uri, Protocol
        protocol, path = parse_uri(uri)
    except ValueError:
        raise ValueError(f'{uri} is not a valid URI. Passing directories or modules to be dynamically uploaded is only supported at the job level (i.e., passed to `ray.init`).')
    if protocol in Protocol.remote_protocols() and (not path.endswith('.zip')) and (not path.endswith('.whl')):
        raise ValueError('Only .zip or .whl files supported for remote URIs.')