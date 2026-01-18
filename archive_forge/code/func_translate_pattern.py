import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
def translate_pattern(pattern, anchor=1, prefix=None, is_regex=0):
    """Translate a shell-like wildcard pattern to a compiled regular
    expression.  Return the compiled regex.  If 'is_regex' true,
    then 'pattern' is directly compiled to a regex (if it's a string)
    or just returned as-is (assumes it's a regex object).
    """
    if is_regex:
        if isinstance(pattern, str):
            return re.compile(pattern)
        else:
            return pattern
    start, _, end = glob_to_re('_').partition('_')
    if pattern:
        pattern_re = glob_to_re(pattern)
        assert pattern_re.startswith(start) and pattern_re.endswith(end)
    else:
        pattern_re = ''
    if prefix is not None:
        prefix_re = glob_to_re(prefix)
        assert prefix_re.startswith(start) and prefix_re.endswith(end)
        prefix_re = prefix_re[len(start):len(prefix_re) - len(end)]
        sep = os.sep
        if os.sep == '\\':
            sep = '\\\\'
        pattern_re = pattern_re[len(start):len(pattern_re) - len(end)]
        pattern_re = '%s\\A%s%s.*%s%s' % (start, prefix_re, sep, pattern_re, end)
    elif anchor:
        pattern_re = '%s\\A%s' % (start, pattern_re[len(start):])
    return re.compile(pattern_re)