from __future__ import annotations
from typing import TYPE_CHECKING
from argparse import ArgumentParser
from .._utils import get_client, print_model
from .._models import BaseModel
class CLIModelIDArgs(BaseModel):
    id: str