from collections import defaultdict
class NameScope(object):

    def __init__(self):
        self._useset = set([''])
        self._basenamemap = defaultdict(int)

    def is_used(self, name):
        return name in self._useset

    def register(self, name, deduplicate=False):
        if deduplicate:
            name = self.deduplicate(name)
        elif self.is_used(name):
            raise DuplicatedNameError(name)
        self._useset.add(name)
        return name

    def deduplicate(self, name):
        basename = name
        while self.is_used(name):
            ident = self._basenamemap[basename] + 1
            self._basenamemap[basename] = ident
            name = '{0}.{1}'.format(basename, ident)
        return name

    def get_child(self):
        return type(self)(parent=self)