import unittest
class DefaultCache(self.Cache):

    def __missing__(self, key):
        try:
            self[key] = key
        except ValueError:
            pass
        return key