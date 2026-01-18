import functools
import sys
import os
import tokenize
def updatecache(filename, module_globals=None):
    """Update a cache entry and return its list of lines.
    If something's wrong, print a message, discard the cache entry,
    and return an empty list."""
    if filename in cache:
        if len(cache[filename]) != 1:
            cache.pop(filename, None)
    if not filename or (filename.startswith('<') and filename.endswith('>')):
        return []
    fullname = filename
    try:
        stat = os.stat(fullname)
    except OSError:
        basename = filename
        if lazycache(filename, module_globals):
            try:
                data = cache[filename][0]()
            except (ImportError, OSError):
                pass
            else:
                if data is None:
                    return []
                cache[filename] = (len(data), None, [line + '\n' for line in data.splitlines()], fullname)
                return cache[filename][2]
        if os.path.isabs(filename):
            return []
        for dirname in sys.path:
            try:
                fullname = os.path.join(dirname, basename)
            except (TypeError, AttributeError):
                continue
            try:
                stat = os.stat(fullname)
                break
            except OSError:
                pass
        else:
            return []
    try:
        with tokenize.open(fullname) as fp:
            lines = fp.readlines()
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    if lines and (not lines[-1].endswith('\n')):
        lines[-1] += '\n'
    size, mtime = (stat.st_size, stat.st_mtime)
    cache[filename] = (size, mtime, lines, fullname)
    return lines