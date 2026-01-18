import ast
import inspect
import io
import linecache
import re
import sys
import textwrap
import tokenize
import astunparse
import gast
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.util import tf_inspect
Returns the source code of given AST.

  Args:
    node: The code to compile, as an AST object.
    indentation: Unused, deprecated. The returning code will always be indented
      at 4 spaces.
    include_encoding_marker: Bool, whether to include a comment on the first
      line to explicitly specify UTF-8 encoding.

  Returns:
    code: The source code generated from the AST object
    source_mapping: A mapping between the user and AutoGraph generated code.
  