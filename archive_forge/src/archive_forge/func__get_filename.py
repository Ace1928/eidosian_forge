import os.path
import re
from .. import exceptions as exc
def _get_filename(content_disposition):
    for match in _OPTION_HEADER_PIECE_RE.finditer(content_disposition):
        k, v = match.groups()
        if k == 'filename':
            return os.path.split(v)[1]
    return None