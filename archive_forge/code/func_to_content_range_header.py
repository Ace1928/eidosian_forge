from __future__ import annotations
from .. import http
def to_content_range_header(self, length):
    """Converts the object into `Content-Range` HTTP header,
        based on given length
        """
    range = self.range_for_length(length)
    if range is not None:
        return f'{self.units} {range[0]}-{range[1] - 1}/{length}'
    return None