from __future__ import annotations
import ast
import json
import os
from io import StringIO
from sys import version_info
from typing import IO, TYPE_CHECKING, Any, Callable, List, Optional, Type, Union
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_community.tools import BaseTool, Tool
from langchain_community.tools.e2b_data_analysis.unparse import Unparser
@property
def uploaded_files_description(self) -> str:
    if len(self._uploaded_files) == 0:
        return ''
    lines = ['The following files available in the sandbox:']
    for f in self._uploaded_files:
        if f.description == '':
            lines.append(f'- path: `{f.remote_path}`')
        else:
            lines.append(f'- path: `{f.remote_path}` \n description: `{f.description}`')
    return '\n'.join(lines)