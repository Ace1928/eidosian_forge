from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
def _remove_path_dot_segments(path):
    segments = path.split('/')
    output = []
    for segment in segments:
        if segment == '.':
            continue
        elif segment != '..':
            output.append(segment)
        elif output:
            output.pop()
    if path.startswith('/') and (not output or output[0]):
        output.insert(0, '')
    if path.endswith(('/.', '/..')):
        output.append('')
    return '/'.join(output)