from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range

        Compare t1 and t2; return true if token types/text, structure match
        exactly.
        The trees are examined in their entirety so that (A B) does not match
        (A B C) nor (A (B C)).
        