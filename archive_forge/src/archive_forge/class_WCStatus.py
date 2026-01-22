import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class WCStatus:
    attrnames = ('modified', 'added', 'conflict', 'unchanged', 'external', 'deleted', 'prop_modified', 'unknown', 'update_available', 'incomplete', 'kindmismatch', 'ignored', 'locked', 'replaced')

    def __init__(self, wcpath, rev=None, modrev=None, author=None):
        self.wcpath = wcpath
        self.rev = rev
        self.modrev = modrev
        self.author = author
        for name in self.attrnames:
            setattr(self, name, [])

    def allpath(self, sort=True, **kw):
        d = {}
        for name in self.attrnames:
            if name not in kw or kw[name]:
                for path in getattr(self, name):
                    d[path] = 1
        l = d.keys()
        if sort:
            l.sort()
        return l
    _rex_status = re.compile('\\s+(\\d+|-)\\s+(\\S+)\\s+(.+?)\\s{2,}(.*)')

    def fromstring(data, rootwcpath, rev=None, modrev=None, author=None):
        """ return a new WCStatus object from data 's'
        """
        rootstatus = WCStatus(rootwcpath, rev, modrev, author)
        update_rev = None
        for line in data.split('\n'):
            if not line.strip():
                continue
            flags, rest = (line[:8], line[8:])
            c0, c1, c2, c3, c4, c5, x6, c7 = flags
            if c0 in '?XI':
                fn = line.split(None, 1)[1]
                if c0 == '?':
                    wcpath = rootwcpath.join(fn, abs=1)
                    rootstatus.unknown.append(wcpath)
                elif c0 == 'X':
                    wcpath = rootwcpath.__class__(rootwcpath.localpath.join(fn, abs=1), auth=rootwcpath.auth)
                    rootstatus.external.append(wcpath)
                elif c0 == 'I':
                    wcpath = rootwcpath.join(fn, abs=1)
                    rootstatus.ignored.append(wcpath)
                continue
            m = WCStatus._rex_status.match(rest)
            if not m:
                if c7 == '*':
                    fn = rest.strip()
                    wcpath = rootwcpath.join(fn, abs=1)
                    rootstatus.update_available.append(wcpath)
                    continue
                if line.lower().find('against revision:') != -1:
                    update_rev = int(rest.split(':')[1].strip())
                    continue
                if line.lower().find('status on external') > -1:
                    continue
                raise ValueError('could not parse line %r' % line)
            else:
                rev, modrev, author, fn = m.groups()
            wcpath = rootwcpath.join(fn, abs=1)
            if c0 == 'M':
                assert wcpath.check(file=1), "didn't expect a directory with changed content here"
                rootstatus.modified.append(wcpath)
            elif c0 == 'A' or c3 == '+':
                rootstatus.added.append(wcpath)
            elif c0 == 'D':
                rootstatus.deleted.append(wcpath)
            elif c0 == 'C':
                rootstatus.conflict.append(wcpath)
            elif c0 == '~':
                rootstatus.kindmismatch.append(wcpath)
            elif c0 == '!':
                rootstatus.incomplete.append(wcpath)
            elif c0 == 'R':
                rootstatus.replaced.append(wcpath)
            elif not c0.strip():
                rootstatus.unchanged.append(wcpath)
            else:
                raise NotImplementedError('received flag %r' % c0)
            if c1 == 'M':
                rootstatus.prop_modified.append(wcpath)
            if c2 == 'L' or c5 == 'K':
                rootstatus.locked.append(wcpath)
            if c7 == '*':
                rootstatus.update_available.append(wcpath)
            if wcpath == rootwcpath:
                rootstatus.rev = rev
                rootstatus.modrev = modrev
                rootstatus.author = author
                if update_rev:
                    rootstatus.update_rev = update_rev
                continue
        return rootstatus
    fromstring = staticmethod(fromstring)