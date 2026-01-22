import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
class MultipartDecoder(object):
    """

    The ``MultipartDecoder`` object parses the multipart payload of
    a bytestring into a tuple of ``Response``-like ``BodyPart`` objects.

    The basic usage is::

        import requests
        from requests_toolbelt import MultipartDecoder

        response = requests.get(url)
        decoder = MultipartDecoder.from_response(response)
        for part in decoder.parts:
            print(part.headers['content-type'])

    If the multipart content is not from a response, basic usage is::

        from requests_toolbelt import MultipartDecoder

        decoder = MultipartDecoder(content, content_type)
        for part in decoder.parts:
            print(part.headers['content-type'])

    For both these usages, there is an optional ``encoding`` parameter. This is
    a string, which is the name of the unicode codec to use (default is
    ``'utf-8'``).

    """

    def __init__(self, content, content_type, encoding='utf-8'):
        self.content_type = content_type
        self.encoding = encoding
        self.parts = tuple()
        self._find_boundary()
        self._parse_body(content)

    def _find_boundary(self):
        ct_info = tuple((x.strip() for x in self.content_type.split(';')))
        mimetype = ct_info[0]
        if mimetype.split('/')[0].lower() != 'multipart':
            raise NonMultipartContentTypeException("Unexpected mimetype in content-type: '{}'".format(mimetype))
        for item in ct_info[1:]:
            attr, value = _split_on_find(item, '=')
            if attr.lower() == 'boundary':
                self.boundary = encode_with(value.strip('"'), self.encoding)

    @staticmethod
    def _fix_first_part(part, boundary_marker):
        bm_len = len(boundary_marker)
        if boundary_marker == part[:bm_len]:
            return part[bm_len:]
        else:
            return part

    def _parse_body(self, content):
        boundary = b''.join((b'--', self.boundary))

        def body_part(part):
            fixed = MultipartDecoder._fix_first_part(part, boundary)
            return BodyPart(fixed, self.encoding)

        def test_part(part):
            return part != b'' and part != b'\r\n' and (part[:4] != b'--\r\n') and (part != b'--')
        parts = content.split(b''.join((b'\r\n', boundary)))
        self.parts = tuple((body_part(x) for x in parts if test_part(x)))

    @classmethod
    def from_response(cls, response, encoding='utf-8'):
        content = response.content
        content_type = response.headers.get('content-type', None)
        return cls(content, content_type, encoding)