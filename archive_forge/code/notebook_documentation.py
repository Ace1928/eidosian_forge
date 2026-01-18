from __future__ import annotations
import logging # isort:skip
import re
from types import ModuleType
from ...core.types import PathLike
from ...util.dependencies import import_required
from .code import CodeHandler

                Given the source of a cell, filter out all cell and line magics.
                