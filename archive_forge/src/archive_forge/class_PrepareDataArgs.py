from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from argparse import ArgumentParser
from .._models import BaseModel
from ...lib._validators import (
class PrepareDataArgs(BaseModel):
    file: str
    quiet: bool