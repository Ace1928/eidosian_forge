import os
import traceback
from io import StringIO
from twisted import copyright
from twisted.python.compat import execfile, networkString
from twisted.python.filepath import _coerceToFilesystemEncoding
from twisted.web import http, resource, server, static, util
import mygreatresource
def ResourceTemplate(path, registry):
    from quixote import ptl_compile
    glob = {'__file__': _coerceToFilesystemEncoding('', path), 'resource': resource._UnsafeErrorPage(500, 'Whoops! Internal Error', rpyNoResource), 'registry': registry}
    with open(path) as f:
        e = ptl_compile.compile_template(f, path)
    code = compile(e, '<source>', 'exec')
    eval(code, glob, glob)
    return glob['resource']