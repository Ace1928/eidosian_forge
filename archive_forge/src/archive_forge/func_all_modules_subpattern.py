from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from libfuturize.fixer_util import touch_import_top
from_import = u"from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
from_import_rename = u"from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"
def all_modules_subpattern():
    u"""
    Builds a pattern for all toplevel names
    (urllib, http, etc)
    """
    names_dot_attrs = [mod.split(u'.') for mod in MAPPING]
    ret = u'( ' + u' | '.join([dotted_name % (simple_name % mod[0], simple_attr % mod[1]) for mod in names_dot_attrs])
    ret += u' | '
    ret += u' | '.join([simple_name % mod[0] for mod in names_dot_attrs if mod[1] == u'__init__']) + u' )'
    return ret