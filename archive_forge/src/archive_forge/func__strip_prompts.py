import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
def _strip_prompts(prompt_re, initial_re=None, turnoff_re=None):
    """Remove matching input prompts from a block of input.

    Parameters
    ----------
    prompt_re : regular expression
        A regular expression matching any input prompt (including continuation)
    initial_re : regular expression, optional
        A regular expression matching only the initial prompt, but not continuation.
        If no initial expression is given, prompt_re will be used everywhere.
        Used mainly for plain Python prompts, where the continuation prompt
        ``...`` is a valid Python expression in Python 3, so shouldn't be stripped.

    Notes
    -----
    If `initial_re` and `prompt_re differ`,
    only `initial_re` will be tested against the first line.
    If any prompt is found on the first two lines,
    prompts will be stripped from the rest of the block.
    """
    if initial_re is None:
        initial_re = prompt_re
    line = ''
    while True:
        line = (yield line)
        if line is None:
            continue
        out, n1 = initial_re.subn('', line, count=1)
        if turnoff_re and (not n1):
            if turnoff_re.match(line):
                while line is not None:
                    line = (yield line)
                continue
        line = (yield out)
        if line is None:
            continue
        out, n2 = prompt_re.subn('', line, count=1)
        line = (yield out)
        if n1 or n2:
            while line is not None:
                line = (yield prompt_re.sub('', line, count=1))
        else:
            while line is not None:
                line = (yield line)