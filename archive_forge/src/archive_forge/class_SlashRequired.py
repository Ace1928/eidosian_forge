from __future__ import annotations
import re
import typing as t
from dataclasses import dataclass
from dataclasses import field
from .converters import ValidationError
from .exceptions import NoMatch
from .exceptions import RequestAliasRedirect
from .exceptions import RequestPath
from .rules import Rule
from .rules import RulePart
class SlashRequired(Exception):
    pass