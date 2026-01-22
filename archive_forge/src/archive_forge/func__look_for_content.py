from io import BytesIO
from testtools import content, content_type
from testtools.compat import _b
from subunit import chunked
def _look_for_content(self, line):
    if line == end_marker:
        self._state.endDetails()
        return
    field, value = line[:-1].decode('utf8').split(' ', 1)
    try:
        main, sub = value.split('/')
    except ValueError:
        raise ValueError('Invalid MIME type %r' % value)
    self._content_type = content_type.ContentType(main, sub)
    self._parse_state = self._get_name