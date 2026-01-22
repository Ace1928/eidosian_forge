import base64
import json
import logging
from itertools import tee
import jmespath
from botocore.exceptions import PaginationError
from botocore.utils import merge_dicts, set_value_from_jmespath
def _inject_starting_params(self, op_kwargs):
    if self._starting_token is not None:
        next_token = self._parse_starting_token()[0]
        self._inject_token_into_kwargs(op_kwargs, next_token)
    if self._page_size is not None:
        op_kwargs[self._limit_key] = self._page_size