from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def _generate_tokens(source, ignore_error_token=False):
    token_generator = tokenize.generate_tokens(StringIO(source).readline)
    try:
        for tok in token_generator:
            yield Token(*tok)
    except tokenize.TokenError:
        if not ignore_error_token:
            raise