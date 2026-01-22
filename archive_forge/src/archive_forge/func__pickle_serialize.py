import marshal
import pickle
from os import PathLike
from pathlib import Path
from typing import Union
from queuelib import queue
from scrapy.utils.request import request_from_dict
def _pickle_serialize(obj):
    try:
        return pickle.dumps(obj, protocol=4)
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        raise ValueError(str(e)) from e