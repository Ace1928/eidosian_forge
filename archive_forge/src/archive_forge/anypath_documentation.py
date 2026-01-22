import os
from abc import ABC
from pathlib import Path
from typing import Any, Union
from .cloudpath import InvalidPrefixError, CloudPath
from .exceptions import AnyPathTypeError
Used as a Pydantic validator. See
        https://pydantic-docs.helpmanual.io/usage/types/#custom-data-types