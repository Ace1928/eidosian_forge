import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
class SvnPathBase(common.PathBase):
    """ Base implementation for SvnPath implementations. """
    sep = '/'

    def _geturl(self):
        return self.strpath
    url = property(_geturl, None, None, 'url of this svn-path.')

    def __str__(self):
        """ return a string representation (including rev-number) """
        return self.strpath

    def __hash__(self):
        return hash(self.strpath)

    def new(self, **kw):
        """ create a modified version of this path. A 'rev' argument
            indicates a new revision.
            the following keyword arguments modify various path parts::

              http://host.com/repo/path/file.ext
              |-----------------------|          dirname
                                        |------| basename
                                        |--|     purebasename
                                            |--| ext
        """
        obj = object.__new__(self.__class__)
        obj.rev = kw.get('rev', self.rev)
        obj.auth = kw.get('auth', self.auth)
        dirname, basename, purebasename, ext = self._getbyspec('dirname,basename,purebasename,ext')
        if 'basename' in kw:
            if 'purebasename' in kw or 'ext' in kw:
                raise ValueError('invalid specification %r' % kw)
        else:
            pb = kw.setdefault('purebasename', purebasename)
            ext = kw.setdefault('ext', ext)
            if ext and (not ext.startswith('.')):
                ext = '.' + ext
            kw['basename'] = pb + ext
        kw.setdefault('dirname', dirname)
        kw.setdefault('sep', self.sep)
        if kw['basename']:
            obj.strpath = '%(dirname)s%(sep)s%(basename)s' % kw
        else:
            obj.strpath = '%(dirname)s' % kw
        return obj

    def _getbyspec(self, spec):
        """ get specified parts of the path.  'arg' is a string
            with comma separated path parts. The parts are returned
            in exactly the order of the specification.

            you may specify the following parts:

            http://host.com/repo/path/file.ext
            |-----------------------|          dirname
                                      |------| basename
                                      |--|     purebasename
                                          |--| ext
        """
        res = []
        parts = self.strpath.split(self.sep)
        for name in spec.split(','):
            name = name.strip()
            if name == 'dirname':
                res.append(self.sep.join(parts[:-1]))
            elif name == 'basename':
                res.append(parts[-1])
            else:
                basename = parts[-1]
                i = basename.rfind('.')
                if i == -1:
                    purebasename, ext = (basename, '')
                else:
                    purebasename, ext = (basename[:i], basename[i:])
                if name == 'purebasename':
                    res.append(purebasename)
                elif name == 'ext':
                    res.append(ext)
                else:
                    raise NameError("Don't know part %r" % name)
        return res

    def __eq__(self, other):
        """ return true if path and rev attributes each match """
        return str(self) == str(other) and (self.rev == other.rev or self.rev == other.rev)

    def __ne__(self, other):
        return not self == other

    def join(self, *args):
        """ return a new Path (with the same revision) which is composed
            of the self Path followed by 'args' path components.
        """
        if not args:
            return self
        args = tuple([arg.strip(self.sep) for arg in args])
        parts = (self.strpath,) + args
        newpath = self.__class__(self.sep.join(parts), self.rev, self.auth)
        return newpath

    def propget(self, name):
        """ return the content of the given property. """
        value = self._propget(name)
        return value

    def proplist(self):
        """ list all property names. """
        content = self._proplist()
        return content

    def size(self):
        """ Return the size of the file content of the Path. """
        return self.info().size

    def mtime(self):
        """ Return the last modification time of the file. """
        return self.info().mtime

    def _escape(self, cmd):
        return _escape_helper(cmd)

    class Checkers(common.Checkers):

        def dir(self):
            try:
                return self.path.info().kind == 'dir'
            except py.error.Error:
                return self._listdirworks()

        def _listdirworks(self):
            try:
                self.path.listdir()
            except py.error.ENOENT:
                return False
            else:
                return True

        def file(self):
            try:
                return self.path.info().kind == 'file'
            except py.error.ENOENT:
                return False

        def exists(self):
            try:
                return self.path.info()
            except py.error.ENOENT:
                return self._listdirworks()