from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from libfuturize.fixer_util import touch_import_top
from_import = u"from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
from_import_rename = u"from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"
class FixImports2(fixer_base.BaseFix):
    run_order = 4
    PATTERN = u' | \n'.join(build_import_pattern(MAPPING, PY2MODULES))

    def transform(self, node, results):
        touch_import_top(u'future', u'standard_library', node)