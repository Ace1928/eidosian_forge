import os
import sys
import stat
import fnmatch
import collections
import errno
def which(cmd, mode=os.F_OK | os.X_OK, path=None):
    """Given a command, mode, and a PATH string, return the path which
    conforms to the given mode on the PATH, or None if there is no such
    file.

    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
    of os.environ.get("PATH"), or can be overridden with a custom search
    path.

    """
    if os.path.dirname(cmd):
        if _access_check(cmd, mode):
            return cmd
        return None
    use_bytes = isinstance(cmd, bytes)
    if path is None:
        path = os.environ.get('PATH', None)
        if path is None:
            try:
                path = os.confstr('CS_PATH')
            except (AttributeError, ValueError):
                path = os.defpath
    if not path:
        return None
    if use_bytes:
        path = os.fsencode(path)
        path = path.split(os.fsencode(os.pathsep))
    else:
        path = os.fsdecode(path)
        path = path.split(os.pathsep)
    if sys.platform == 'win32':
        curdir = os.curdir
        if use_bytes:
            curdir = os.fsencode(curdir)
        if curdir not in path:
            path.insert(0, curdir)
        pathext_source = os.getenv('PATHEXT') or _WIN_DEFAULT_PATHEXT
        pathext = [ext for ext in pathext_source.split(os.pathsep) if ext]
        if use_bytes:
            pathext = [os.fsencode(ext) for ext in pathext]
        if any((cmd.lower().endswith(ext.lower()) for ext in pathext)):
            files = [cmd]
        else:
            files = [cmd + ext for ext in pathext]
    else:
        files = [cmd]
    seen = set()
    for dir in path:
        normdir = os.path.normcase(dir)
        if not normdir in seen:
            seen.add(normdir)
            for thefile in files:
                name = os.path.join(dir, thefile)
                if _access_check(name, mode):
                    return name
    return None