import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def _make_multipart(self, subtype, disallowed_subtypes, boundary):
    if self.get_content_maintype() == 'multipart':
        existing_subtype = self.get_content_subtype()
        disallowed_subtypes = disallowed_subtypes + (subtype,)
        if existing_subtype in disallowed_subtypes:
            raise ValueError('Cannot convert {} to {}'.format(existing_subtype, subtype))
    keep_headers = []
    part_headers = []
    for name, value in self._headers:
        if name.lower().startswith('content-'):
            part_headers.append((name, value))
        else:
            keep_headers.append((name, value))
    if part_headers:
        part = type(self)(policy=self.policy)
        part._headers = part_headers
        part._payload = self._payload
        self._payload = [part]
    else:
        self._payload = []
    self._headers = keep_headers
    self['Content-Type'] = 'multipart/' + subtype
    if boundary is not None:
        self.set_param('boundary', boundary)