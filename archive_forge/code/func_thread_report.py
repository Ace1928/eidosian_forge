import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
def thread_report(path=SCRIPT_NAME + '/hello', concurrency=safe_threads):
    sess = ABSession(path)
    attrs, names, patterns = list(zip(*sess.parse_patterns))
    avg = dict.fromkeys(attrs, 0.0)
    yield (('threads',) + names)
    for c in concurrency:
        sess.concurrency = c
        sess.run()
        row = [c]
        for attr in attrs:
            val = getattr(sess, attr)
            if val is None:
                print(sess.output)
                row = None
                break
            val = float(val)
            avg[attr] += float(val)
            row.append(val)
        if row:
            yield row
    yield (['Average'] + [str(avg[attr] / len(concurrency)) for attr in attrs])