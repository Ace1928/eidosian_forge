import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _get_non_aggregate_keys(self, config):
    keys = []
    for key in config.get('non_aggregate_keys', []):
        keys.append(jmespath.compile(key))
    return keys