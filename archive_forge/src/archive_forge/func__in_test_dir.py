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
def _in_test_dir():
    """Check if the current working directory ends with gradio/js/preview/test."""
    return Path.cwd().parts[-4:] == ('gradio', 'js', 'preview', 'test')