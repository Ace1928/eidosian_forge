from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast
from argparse import ArgumentParser
from .._utils import get_client, print_model
from .._models import BaseModel
from .._progress import BufferReader
class CLIFileCreateArgs(BaseModel):
    file: str
    purpose: str