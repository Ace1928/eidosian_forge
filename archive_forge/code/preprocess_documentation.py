from codeop import CommandCompiler
from typing import Match
from itertools import tee, islice, chain
from ..lazyre import LazyReCompile
Indents blank lines that would otherwise cause early compilation

    Only really works if starting on a new line