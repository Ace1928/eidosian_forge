import sys
import os
import builtins
import _sitebuiltins
import io
import stat
def addpackage(sitedir, name, known_paths):
    """Process a .pth file within the site-packages directory:
       For each line in the file, either combine it with sitedir to a path
       and add that to known_paths, or execute it if it starts with 'import '.
    """
    if known_paths is None:
        known_paths = _init_pathinfo()
        reset = True
    else:
        reset = False
    fullname = os.path.join(sitedir, name)
    try:
        st = os.lstat(fullname)
    except OSError:
        return
    if getattr(st, 'st_flags', 0) & stat.UF_HIDDEN or getattr(st, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_HIDDEN:
        _trace(f'Skipping hidden .pth file: {fullname!r}')
        return
    _trace(f'Processing .pth file: {fullname!r}')
    try:
        f = io.TextIOWrapper(io.open_code(fullname), encoding='locale')
    except OSError:
        return
    with f:
        for n, line in enumerate(f):
            if line.startswith('#'):
                continue
            if line.strip() == '':
                continue
            try:
                if line.startswith(('import ', 'import\t')):
                    exec(line)
                    continue
                line = line.rstrip()
                dir, dircase = makepath(sitedir, line)
                if not dircase in known_paths and os.path.exists(dir):
                    sys.path.append(dir)
                    known_paths.add(dircase)
            except Exception:
                print('Error processing line {:d} of {}:\n'.format(n + 1, fullname), file=sys.stderr)
                import traceback
                for record in traceback.format_exception(*sys.exc_info()):
                    for line in record.splitlines():
                        print('  ' + line, file=sys.stderr)
                print('\nRemainder of file ignored', file=sys.stderr)
                break
    if reset:
        known_paths = None
    return known_paths