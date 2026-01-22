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
def _render_endpoints_info(self, name_or_index: str | int, endpoints_info: dict[str, list[ParameterInfo]]) -> str:
    parameter_info = endpoints_info['parameters']
    parameter_names = [p.get('parameter_name') or p['label'] for p in parameter_info]
    parameter_names = [utils.sanitize_parameter_names(p) for p in parameter_names]
    rendered_parameters = ', '.join(parameter_names)
    if rendered_parameters:
        rendered_parameters = rendered_parameters + ', '
    return_values = [p['label'] for p in endpoints_info['returns']]
    return_values = [utils.sanitize_parameter_names(r) for r in return_values]
    rendered_return_values = ', '.join(return_values)
    if len(return_values) > 1:
        rendered_return_values = f'({rendered_return_values})'
    if isinstance(name_or_index, str):
        final_param = f'api_name="{name_or_index}"'
    elif isinstance(name_or_index, int):
        final_param = f'fn_index={name_or_index}'
    else:
        raise ValueError('name_or_index must be a string or integer')
    human_info = f'\n - predict({rendered_parameters}{final_param}) -> {rendered_return_values}\n'
    human_info += '    Parameters:\n'
    if parameter_info:
        for info in parameter_info:
            desc = f' ({info['python_type']['description']})' if info['python_type'].get('description') else ''
            default_value = info.get('parameter_default')
            default_value = utils.traverse(default_value, lambda x: f'file("{x['url']}")', utils.is_file_obj_with_meta)
            default_info = '(required)' if not info.get('parameter_has_default', False) else f'(not required, defaults to:   {default_value})'
            type_ = info['python_type']['type']
            if info.get('parameter_has_default', False) and default_value is None:
                type_ += ' | None'
            human_info += f'     - [{info['component']}] {utils.sanitize_parameter_names(info.get('parameter_name') or info['label'])}: {type_} {default_info} {desc} \n'
    else:
        human_info += '     - None\n'
    human_info += '    Returns:\n'
    if endpoints_info['returns']:
        for info in endpoints_info['returns']:
            desc = f' ({info['python_type']['description']})' if info['python_type'].get('description') else ''
            type_ = info['python_type']['type']
            human_info += f'     - [{info['component']}] {utils.sanitize_parameter_names(info['label'])}: {type_}{desc} \n'
    else:
        human_info += '     - None\n'
    return human_info