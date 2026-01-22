from typing import Dict, Type, Callable, List
class Bencached(object):
    __slots__ = ['bencoded']

    def __init__(self, s):
        self.bencoded = s