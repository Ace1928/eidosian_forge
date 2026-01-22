from __future__ import annotations
import csv
import datetime
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any
import filelock
import huggingface_hub
from gradio_client import utils as client_utils
from gradio_client.documentation import document
import gradio as gr
from gradio import utils
@staticmethod
def _save_as_jsonl(data_file: Path, headers: list[str], row: list[Any]) -> str:
    """Save data as JSONL and return the sample name (uuid)."""
    Path.mkdir(data_file.parent, parents=True, exist_ok=True)
    with open(data_file, 'w') as f:
        json.dump(dict(zip(headers, row)), f)
    return data_file.parent.name