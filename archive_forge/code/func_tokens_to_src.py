from __future__ import annotations
import argparse
import io
import keyword
import re
import sys
import tokenize
from typing import Generator
from typing import Iterable
from typing import NamedTuple
from typing import Pattern
from typing import Sequence
def tokens_to_src(tokens: Iterable[Token]) -> str:
    return ''.join((tok.src for tok in tokens))