import os
class _NoModify(_Rule):
    name = 'nomodify'

    def __init__(self, path):
        self.path = path

    def fix(self, path):
        pass