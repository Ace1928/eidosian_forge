import abc
import os  # noqa
from typing import (
import attr
from . import hdrs
from .abc import AbstractView
from .typedefs import Handler, PathLike
Route definition table