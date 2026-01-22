import re
import sys
import cgi
import os
import os.path
import urllib.parse
import cherrypy
def _show_branch(root, base, path, pct=0, showpct=False, exclude='', coverage=the_coverage):
    dirs = [k for k, v in root.items() if v]
    dirs.sort()
    for name in dirs:
        newpath = os.path.join(path, name)
        if newpath.lower().startswith(base):
            relpath = newpath[len(base):]
            yield ('| ' * relpath.count(os.sep))
            yield ("<a class='directory' href='menu?base=%s&exclude=%s'>%s</a>\n" % (newpath, urllib.parse.quote_plus(exclude), name))
        for chunk in _show_branch(root[name], base, newpath, pct, showpct, exclude, coverage=coverage):
            yield chunk
    if path.lower().startswith(base):
        relpath = path[len(base):]
        files = [k for k, v in root.items() if not v]
        files.sort()
        for name in files:
            newpath = os.path.join(path, name)
            pc_str = ''
            if showpct:
                try:
                    _, statements, _, missing, _ = coverage.analysis2(newpath)
                except Exception:
                    pass
                else:
                    pc = _percent(statements, missing)
                    pc_str = ('%3d%% ' % pc).replace(' ', '&nbsp;')
                    if pc < float(pct) or pc == -1:
                        pc_str = "<span class='fail'>%s</span>" % pc_str
                    else:
                        pc_str = "<span class='pass'>%s</span>" % pc_str
            yield (TEMPLATE_ITEM % ('| ' * (relpath.count(os.sep) + 1), pc_str, newpath, name))