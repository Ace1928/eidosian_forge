import re
from . import errors, osutils, transport
def _deserialize_view_content(self, view_content):
    """Convert a stream into view keywords and a dictionary of views."""
    if view_content == b'':
        return ({}, {})
    lines = view_content.splitlines()
    match = _VIEWS_FORMAT_MARKER_RE.match(lines[0])
    if not match:
        raise ValueError('format marker missing from top of views file')
    elif match.group(1) != b'1':
        raise ValueError('cannot decode views format %s' % match.group(1))
    try:
        keywords = {}
        views = {}
        in_views = False
        for line in lines[1:]:
            text = line.decode('utf-8')
            if in_views:
                parts = text.split('\x00')
                view = parts.pop(0)
                views[view] = parts
            elif text == 'views:':
                in_views = True
                continue
            elif text.find('=') >= 0:
                keyword, value = text.split('=', 1)
                keywords[keyword] = value
            else:
                raise ValueError('failed to deserialize views line %s', text)
        return (keywords, views)
    except ValueError as e:
        raise ValueError('failed to deserialize views content %r: %s' % (view_content, e))