import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union
from requests import get
from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params
def _interactive_request_single(node, param):
    """Prompt the user to select a single option from a list."""
    options = list(node)
    if len(options) == 1:
        print(f'Using {options[0]} as it is the only {param} available.')
        sleep(1)
        return options[0]
    print(f'Please select a {param}:')
    print('\n'.join((f'\t{i + 1}) {option}' for i, option in enumerate(options))))
    try:
        choice = int(input(f'Choice [1-{len(options)}]: '))
    except ValueError as e:
        raise ValueError(f'Must enter an integer between 1 and {len(options)}') from e
    if choice < 1 or choice > len(options):
        raise ValueError(f'Must enter an integer between 1 and {len(options)}')
    return options[choice - 1]