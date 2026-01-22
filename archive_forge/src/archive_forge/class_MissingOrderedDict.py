import threading
import json
from gc import get_objects, garbage
from kivy.clock import Clock
from kivy.cache import Cache
from collections import OrderedDict
from kivy.logger import Logger
class MissingOrderedDict(OrderedDict):

    def __missing__(self, key):
        self[key] = [0] * history_max
        return self[key]