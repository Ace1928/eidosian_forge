from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
def DefaultParseValue(value):
    """The default argument parsing function used by Fire CLIs.

  If the value is made of only Python literals and containers, then the value
  is parsed as it's Python value. Otherwise, provided the value contains no
  quote, escape, or parenthetical characters, the value is treated as a string.

  Args:
    value: A string from the command line to be parsed for use in a Fire CLI.
  Returns:
    The parsed value, of the type determined most appropriate.
  """
    try:
        return _LiteralEval(value)
    except (SyntaxError, ValueError):
        return value