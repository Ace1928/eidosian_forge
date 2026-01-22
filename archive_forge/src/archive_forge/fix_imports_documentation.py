from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin, Newline, does_tree_import
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.pytree import Node, Leaf
from libfuturize.fixer_util import touch_import_top
from_import_match = u"from_import=import_from< 'from' %s 'import' imported=any >"
from_import_submod_match = u"from_import_submod=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* > ) >"

    Accepts a string and returns a pattern of possible patterns involving that name
    Called by simple_mapping_to_pattern for each name in the mapping it receives.
    