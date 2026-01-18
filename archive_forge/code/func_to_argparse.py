from __future__ import annotations
import argparse
import enum
import functools
import logging
from typing import Any
from typing import Callable
from typing import Sequence
from flake8 import utils
from flake8.plugins.finder import Plugins
def to_argparse(self) -> tuple[list[str], dict[str, Any]]:
    """Convert a Flake8 Option to argparse ``add_argument`` arguments."""
    return (self.option_args, self.filtered_option_kwargs)