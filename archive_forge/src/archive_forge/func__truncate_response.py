import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _truncate_response(self, parsed, primary_result_key, truncate_amount, starting_truncation, next_token):
    original = primary_result_key.search(parsed)
    if original is None:
        original = []
    amount_to_keep = len(original) - truncate_amount
    truncated = original[:amount_to_keep]
    set_value_from_jmespath(parsed, primary_result_key.expression, truncated)
    next_token['boto_truncate_amount'] = amount_to_keep + starting_truncation
    self.resume_token = next_token