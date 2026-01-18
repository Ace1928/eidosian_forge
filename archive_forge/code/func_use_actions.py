import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
@classmethod
def use_actions(cls, actions):
    cls._actions = actions