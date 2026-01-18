import re
import shlex
from typing import List
from mlflow.utils.os import is_windows
def truncate_str_from_middle(s, max_length):
    assert max_length > 5
    if len(s) <= max_length:
        return s
    else:
        left_part_len = (max_length - 3) // 2
        right_part_len = max_length - 3 - left_part_len
        return f'{s[:left_part_len]}...{s[-right_part_len:]}'