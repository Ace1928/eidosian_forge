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
def _interactive_request_attributes(options):
    """Prompt the user to select a list of attributes."""
    prompt = 'Please select attributes:'
    for i, option in enumerate(options):
        if option == 'full':
            option = 'full (all attributes)'
        prompt += f'\n\t{i + 1}) {option}'
    print(prompt)
    choices = input(f'Choice (comma-separated list of options) [1-{len(options)}]: ').split(',')
    try:
        choices = list(map(int, choices))
    except ValueError as e:
        raise ValueError(f'Must enter a list of integers between 1 and {len(options)}') from e
    if any((choice < 1 or choice > len(options) for choice in choices)):
        raise ValueError(f'Must enter a list of integers between 1 and {len(options)}')
    return [options[choice - 1] for choice in choices]