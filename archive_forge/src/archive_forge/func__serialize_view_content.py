import re
from . import errors, osutils, transport
def _serialize_view_content(self, keywords, view_dict):
    """Convert view keywords and a view dictionary into a stream."""
    lines = [_VIEWS_FORMAT1_MARKER]
    for key in keywords:
        line = '{}={}\n'.format(key, keywords[key])
        lines.append(line.encode('utf-8'))
    if view_dict:
        lines.append(b'views:\n')
        for view in sorted(view_dict):
            view_data = '{}\x00{}\n'.format(view, '\x00'.join(view_dict[view]))
            lines.append(view_data.encode('utf-8'))
    return b''.join(lines)