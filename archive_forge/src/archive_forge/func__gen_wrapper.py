from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import ast
import contextlib
import functools
import itertools
import six
from six.moves import zip
import sys
from pasta.base import ast_constants
from pasta.base import ast_utils
from pasta.base import formatting as fmt
from pasta.base import token_generator
def _gen_wrapper(f, scope=True, prefix=True, suffix=True, max_suffix_lines=None, semicolon=False, comment=False, statement=False):

    @contextlib.wraps(f)
    def wrapped(self, node, *args, **kwargs):
        with self.scope(node, trailing_comma=False) if scope else _noop_context():
            if prefix:
                self.prefix(node, default=self._indent if statement else '')
            f(self, node, *args, **kwargs)
            if suffix:
                self.suffix(node, max_lines=max_suffix_lines, semicolon=semicolon, comment=comment, default='\n' if statement else '')
    return wrapped