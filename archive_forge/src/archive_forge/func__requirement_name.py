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
def _requirement_name(self) -> str:
    """Return a description of the requirement that triggered me.

        This default implementation returns long description of the req, with
        line numbers

        """
    return str(self.req) if self.req else 'unknown package'