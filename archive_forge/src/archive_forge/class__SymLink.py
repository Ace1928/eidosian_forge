import os
class _SymLink(_Rule):
    name = 'symlink'
    inherit = True

    def __init__(self, path, dest):
        self.path = path
        self.dest = dest

    def check(self, path):
        assert path == self.path, '_Symlink should only be passed specific path %s (not %s)' % (self.path, path)
        try:
            link = os.path.readlink(path)
        except OSError as e:
            if e.errno != 22:
                raise
            return ['Path %s is not a symlink (should point to %s)' % (path, self.dest)]
        if link != self.dest:
            return ['Path %s should symlink to %s, not %s' % (path, self.dest, link)]
        return []

    def fix(self, path):
        assert path == self.path, '_Symlink should only be passed specific path %s (not %s)' % (self.path, path)
        if not os.path.exists(path):
            os.symlink(path, self.dest)
        else:
            print('Not symlinking %s' % path)