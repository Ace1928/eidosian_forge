from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from ..common.utils import isStrSpace
from ..ruler import StateBase
from ..token import Token
from ..utils import EnvType
Check if line is a code block,
        i.e. the code block rule is enabled and text is indented by more than 3 spaces.
        