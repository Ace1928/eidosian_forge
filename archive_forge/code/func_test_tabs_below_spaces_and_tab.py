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
@test_utils.requires_features('mixed_tabs_spaces')
def test_tabs_below_spaces_and_tab(self):
    for num_spaces in range(1, 8):
        t = pasta.parse(textwrap.dedent('          if a:\n          {WS}{ONETAB}if b:\n          {ONETAB}{ONETAB}c\n          ').format(ONETAB='\t', WS=' ' * num_spaces))
        node_c = t.body[0].body[0].body[0]
        self.assertEqual(fmt.get(node_c, 'indent_diff'), '\t')