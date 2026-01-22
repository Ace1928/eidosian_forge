import logging
import re
from .compat import string_types
from .util import parse_requirement
def _suggest_normalized_version(s):
    """Suggest a normalized version close to the given version string.

    If you have a version string that isn't rational (i.e. NormalizedVersion
    doesn't like it) then you might be able to get an equivalent (or close)
    rational version from this function.

    This does a number of simple normalizations to the given string, based
    on observation of versions currently in use on PyPI. Given a dump of
    those version during PyCon 2009, 4287 of them:
    - 2312 (53.93%) match NormalizedVersion without change
      with the automatic suggestion
    - 3474 (81.04%) match when using this suggestion method

    @param s {str} An irrational version string.
    @returns A rational version string, or None, if couldn't determine one.
    """
    try:
        _normalized_key(s)
        return s
    except UnsupportedVersionError:
        pass
    rs = s.lower()
    for orig, repl in (('-alpha', 'a'), ('-beta', 'b'), ('alpha', 'a'), ('beta', 'b'), ('rc', 'c'), ('-final', ''), ('-pre', 'c'), ('-release', ''), ('.release', ''), ('-stable', ''), ('+', '.'), ('_', '.'), (' ', ''), ('.final', ''), ('final', '')):
        rs = rs.replace(orig, repl)
    rs = re.sub('pre$', 'pre0', rs)
    rs = re.sub('dev$', 'dev0', rs)
    rs = re.sub('([abc]|rc)[\\-\\.](\\d+)$', '\\1\\2', rs)
    rs = re.sub('[\\-\\.](dev)[\\-\\.]?r?(\\d+)$', '.\\1\\2', rs)
    rs = re.sub('[.~]?([abc])\\.?', '\\1', rs)
    if rs.startswith('v'):
        rs = rs[1:]
    rs = re.sub('\\b0+(\\d+)(?!\\d)', '\\1', rs)
    rs = re.sub('(\\d+[abc])$', '\\g<1>0', rs)
    rs = re.sub('\\.?(dev-r|dev\\.r)\\.?(\\d+)$', '.dev\\2', rs)
    rs = re.sub('-(a|b|c)(\\d+)$', '\\1\\2', rs)
    rs = re.sub('[\\.\\-](dev|devel)$', '.dev0', rs)
    rs = re.sub('(?![\\.\\-])dev$', '.dev0', rs)
    rs = re.sub('(final|stable)$', '', rs)
    rs = re.sub('\\.?(r|-|-r)\\.?(\\d+)$', '.post\\2', rs)
    rs = re.sub('\\.?(dev|git|bzr)\\.?(\\d+)$', '.dev\\2', rs)
    rs = re.sub('\\.?(pre|preview|-c)(\\d+)$', 'c\\g<2>', rs)
    rs = re.sub('p(\\d+)$', '.post\\1', rs)
    try:
        _normalized_key(rs)
    except UnsupportedVersionError:
        rs = None
    return rs