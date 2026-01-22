import os, re
import fnmatch
import functools
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsInternalError
from distutils import log
def _parse_template_line(self, line):
    words = line.split()
    action = words[0]
    patterns = dir = dir_pattern = None
    if action in ('include', 'exclude', 'global-include', 'global-exclude'):
        if len(words) < 2:
            raise DistutilsTemplateError("'%s' expects <pattern1> <pattern2> ..." % action)
        patterns = [convert_path(w) for w in words[1:]]
    elif action in ('recursive-include', 'recursive-exclude'):
        if len(words) < 3:
            raise DistutilsTemplateError("'%s' expects <dir> <pattern1> <pattern2> ..." % action)
        dir = convert_path(words[1])
        patterns = [convert_path(w) for w in words[2:]]
    elif action in ('graft', 'prune'):
        if len(words) != 2:
            raise DistutilsTemplateError("'%s' expects a single <dir_pattern>" % action)
        dir_pattern = convert_path(words[1])
    else:
        raise DistutilsTemplateError("unknown action '%s'" % action)
    return (action, patterns, dir, dir_pattern)