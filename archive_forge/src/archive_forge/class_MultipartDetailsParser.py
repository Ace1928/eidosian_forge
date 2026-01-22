from io import BytesIO
from testtools import content, content_type
from testtools.compat import _b
from subunit import chunked
class MultipartDetailsParser(DetailsParser):
    """Parser for multi-part [] surrounded MIME typed chunked details."""

    def __init__(self, state):
        self._state = state
        self._details = {}
        self._parse_state = self._look_for_content

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

    def _get_name(self, line):
        self._name = line[:-1].decode('utf8')
        self._body = BytesIO()
        self._chunk_parser = chunked.Decoder(self._body)
        self._parse_state = self._feed_chunks

    def _feed_chunks(self, line):
        residue = self._chunk_parser.write(line)
        if residue is not None:
            assert residue == empty, 'residue: %r' % (residue,)
            body = self._body
            self._details[self._name] = content.Content(self._content_type, lambda: [body.getvalue()])
            self._chunk_parser.close()
            self._parse_state = self._look_for_content

    def get_details(self, for_skip=False):
        return self._details

    def get_message(self):
        return None

    def lineReceived(self, line):
        self._parse_state(line)