import random
import time
import unittest
class DummyLRUCache(dict):

    def put(self, k, v):
        return self.__setitem__(k, v)