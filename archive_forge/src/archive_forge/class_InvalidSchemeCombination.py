import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Union
from pip._vendor.requests.models import Request, Response
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text
class InvalidSchemeCombination(InstallationError):

    def __str__(self) -> str:
        before = ', '.join((str(a) for a in self.args[:-1]))
        return f'Cannot set {before} and {self.args[-1]} together'