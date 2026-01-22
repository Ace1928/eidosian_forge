import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _handle_first_request(self, parsed, primary_result_key, starting_truncation):
    starting_truncation = self._parse_starting_token()[1]
    all_data = primary_result_key.search(parsed)
    if isinstance(all_data, (list, str)):
        data = all_data[starting_truncation:]
    else:
        data = None
    set_value_from_jmespath(parsed, primary_result_key.expression, data)
    for token in self.result_keys:
        if token == primary_result_key:
            continue
        sample = token.search(parsed)
        if isinstance(sample, list):
            empty_value = []
        elif isinstance(sample, str):
            empty_value = ''
        elif isinstance(sample, (int, float)):
            empty_value = 0
        else:
            empty_value = None
        set_value_from_jmespath(parsed, token.expression, empty_value)
    return starting_truncation