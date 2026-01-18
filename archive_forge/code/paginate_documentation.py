import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
Create paginator object for an operation.

        This returns an iterable object.  Iterating over
        this object will yield a single page of a response
        at a time.

        