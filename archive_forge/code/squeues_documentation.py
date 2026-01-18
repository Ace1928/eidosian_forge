import marshal
import pickle
from os import PathLike
from pathlib import Path
from typing import Union
from queuelib import queue
from scrapy.utils.request import request_from_dict
Returns the next object to be returned by :meth:`pop`,
            but without removing it from the queue.

            Raises :exc:`NotImplementedError` if the underlying queue class does
            not implement a ``peek`` method, which is optional for queues.
            