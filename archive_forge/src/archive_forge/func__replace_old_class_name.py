from __future__ import annotations
import dataclasses
import inspect
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Literal
import gradio
import gradio as gr
from {package_name} import {name}
import gradio as gr
from {package_name} import {name}
from .{name.lower()} import {name}
def _replace_old_class_name(old_class_name: str, new_class_name: str, content: str):
    pattern = f'(?<=\\b)(?<!\\bimport\\s)(?<!\\.){re.escape(old_class_name)}(?=\\b)'
    return re.sub(pattern, new_class_name, content)