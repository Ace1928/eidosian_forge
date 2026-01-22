from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
class PrefixSuffixGoldenTest(with_metaclass(PrefixSuffixGoldenTestMeta, test_utils.TestCase)):
    """Checks the prefix and suffix on each node in the AST.

  This uses golden files in testdata/ast/golden. To regenerate these files, run
  python setup.py test -s pasta.base.annotate_test.generate_goldens
  """
    maxDiff = None