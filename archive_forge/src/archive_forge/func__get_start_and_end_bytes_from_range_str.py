from typing import Tuple
from libcloud.test import MockHttp  # pylint: disable-msg=E0611
def _get_start_and_end_bytes_from_range_str(self, range_str, body):
    range_str = range_str.split('bytes=')[1]
    range_str = range_str.split('-')
    range_str = [value for value in range_str if value.strip()]
    start_bytes = int(range_str[0])
    if len(range_str) == 2:
        end_bytes = int(range_str[1])
    else:
        end_bytes = len(body)
    return (start_bytes, end_bytes)