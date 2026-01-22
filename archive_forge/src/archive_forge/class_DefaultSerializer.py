import json
import pickle
from functools import partial
from typing import Optional, Type, Union
from .utils import import_attribute
class DefaultSerializer:
    dumps = partial(pickle.dumps, protocol=pickle.HIGHEST_PROTOCOL)
    loads = pickle.loads