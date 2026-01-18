from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def view_api(self, all_endpoints: bool | None=None, print_info: bool=True, return_format: Literal['dict', 'str'] | None=None) -> dict | str | None:
    """
        Prints the usage info for the API. If the Gradio app has multiple API endpoints, the usage info for each endpoint will be printed separately. If return_format="dict" the info is returned in dictionary format, as shown in the example below.

        Parameters:
            all_endpoints: If True, prints information for both named and unnamed endpoints in the Gradio app. If False, will only print info about named endpoints. If None (default), will print info about named endpoints, unless there aren't any -- in which it will print info about unnamed endpoints.
            print_info: If True, prints the usage info to the console. If False, does not print the usage info.
            return_format: If None, nothing is returned. If "str", returns the same string that would be printed to the console. If "dict", returns the usage info as a dictionary that can be programmatically parsed, and *all endpoints are returned in the dictionary* regardless of the value of `all_endpoints`. The format of the dictionary is in the docstring of this method.
        Example:
            from gradio_client import Client
            client = Client(src="gradio/calculator")
            client.view_api(return_format="dict")
            >> {
                'named_endpoints': {
                    '/predict': {
                        'parameters': [
                            {
                                'label': 'num1',
                                'python_type': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                                'example_input': '5'
                            },
                            {
                                'label': 'operation',
                                'python_type': 'str',
                                'type_description': 'string value',
                                'component': 'Radio',
                                'example_input': 'add'
                            },
                            {
                                'label': 'num2',
                                'python_type': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                                'example_input': '5'
                            },
                        ],
                        'returns': [
                            {
                                'label': 'output',
                                'python_type': 'int | float',
                                'type_description': 'numeric value',
                                'component': 'Number',
                            },
                        ]
                    },
                    '/flag': {
                        'parameters': [
                            ...
                            ],
                        'returns': [
                            ...
                            ]
                        }
                    }
                'unnamed_endpoints': {
                    2: {
                        'parameters': [
                            ...
                            ],
                        'returns': [
                            ...
                            ]
                        }
                    }
                }
            }

        """
    num_named_endpoints = len(self._info['named_endpoints'])
    num_unnamed_endpoints = len(self._info['unnamed_endpoints'])
    if num_named_endpoints == 0 and all_endpoints is None:
        all_endpoints = True
    human_info = 'Client.predict() Usage Info\n---------------------------\n'
    human_info += f'Named API endpoints: {num_named_endpoints}\n'
    for api_name, endpoint_info in self._info['named_endpoints'].items():
        human_info += self._render_endpoints_info(api_name, endpoint_info)
    if all_endpoints:
        human_info += f'\nUnnamed API endpoints: {num_unnamed_endpoints}\n'
        for fn_index, endpoint_info in self._info['unnamed_endpoints'].items():
            human_info += self._render_endpoints_info(int(fn_index), endpoint_info)
    elif num_unnamed_endpoints > 0:
        human_info += f'\nUnnamed API endpoints: {num_unnamed_endpoints}, to view, run Client.view_api(all_endpoints=True)\n'
    if print_info:
        print(human_info)
    if return_format == 'str':
        return human_info
    elif return_format == 'dict':
        return self._info