import ast
from codeop import CommandCompiler, Compile
import re
import sys
import tokenize
from typing import List, Tuple, Optional, Any
import warnings
from IPython.utils import tokenutil
class PromptStripper:
    """Remove matching input prompts from a block of input.

    Parameters
    ----------
    prompt_re : regular expression
        A regular expression matching any input prompt (including continuation,
        e.g. ``...``)
    initial_re : regular expression, optional
        A regular expression matching only the initial prompt, but not continuation.
        If no initial expression is given, prompt_re will be used everywhere.
        Used mainly for plain Python prompts (``>>>``), where the continuation prompt
        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

    Notes
    -----

    If initial_re and prompt_re differ,
    only initial_re will be tested against the first line.
    If any prompt is found on the first two lines,
    prompts will be stripped from the rest of the block.
    """

    def __init__(self, prompt_re, initial_re=None):
        self.prompt_re = prompt_re
        self.initial_re = initial_re or prompt_re

    def _strip(self, lines):
        return [self.prompt_re.sub('', l, count=1) for l in lines]

    def __call__(self, lines):
        if not lines:
            return lines
        if self.initial_re.match(lines[0]) or (len(lines) > 1 and self.prompt_re.match(lines[1])):
            return self._strip(lines)
        return lines