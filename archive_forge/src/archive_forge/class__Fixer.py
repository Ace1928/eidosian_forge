import os
class _Fixer(_Strategy):

    def noexists(self, path, checker):
        pass

    def check(self, path, checker):
        checker.fix(path)