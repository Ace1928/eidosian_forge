import sys
def _formatwarnmsg_impl(msg):
    category = msg.category.__name__
    s = f'{msg.filename}:{msg.lineno}: {category}: {msg.message}\n'
    if msg.line is None:
        try:
            import linecache
            line = linecache.getline(msg.filename, msg.lineno)
        except Exception:
            line = None
            linecache = None
    else:
        line = msg.line
    if line:
        line = line.strip()
        s += '  %s\n' % line
    if msg.source is not None:
        try:
            import tracemalloc
        except Exception:
            suggest_tracemalloc = False
            tb = None
        else:
            try:
                suggest_tracemalloc = not tracemalloc.is_tracing()
                tb = tracemalloc.get_object_traceback(msg.source)
            except Exception:
                suggest_tracemalloc = False
                tb = None
        if tb is not None:
            s += 'Object allocated at (most recent call last):\n'
            for frame in tb:
                s += '  File "%s", lineno %s\n' % (frame.filename, frame.lineno)
                try:
                    if linecache is not None:
                        line = linecache.getline(frame.filename, frame.lineno)
                    else:
                        line = None
                except Exception:
                    line = None
                if line:
                    line = line.strip()
                    s += '    %s\n' % line
        elif suggest_tracemalloc:
            s += f'{category}: Enable tracemalloc to get the object allocation traceback\n'
    return s