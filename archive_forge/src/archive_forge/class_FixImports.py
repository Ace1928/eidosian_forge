from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin, Newline, does_tree_import
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.pytree import Node, Leaf
from libfuturize.fixer_util import touch_import_top
from_import_match = u"from_import=import_from< 'from' %s 'import' imported=any >"
from_import_submod_match = u"from_import_submod=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* > ) >"
class FixImports(fixer_base.BaseFix):
    PATTERN = u' | \n'.join([all_patterns(name) for name in MAPPING])
    PATTERN = u' | \n'.join((PATTERN, multiple_name_import_match))

    def transform(self, node, results):
        touch_import_top(u'future', u'standard_library', node)