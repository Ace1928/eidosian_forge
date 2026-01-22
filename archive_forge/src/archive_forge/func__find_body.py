import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def _find_body(self, part, preferencelist):
    if part.is_attachment():
        return
    maintype, subtype = part.get_content_type().split('/')
    if maintype == 'text':
        if subtype in preferencelist:
            yield (preferencelist.index(subtype), part)
        return
    if maintype != 'multipart' or not self.is_multipart():
        return
    if subtype != 'related':
        for subpart in part.iter_parts():
            yield from self._find_body(subpart, preferencelist)
        return
    if 'related' in preferencelist:
        yield (preferencelist.index('related'), part)
    candidate = None
    start = part.get_param('start')
    if start:
        for subpart in part.iter_parts():
            if subpart['content-id'] == start:
                candidate = subpart
                break
    if candidate is None:
        subparts = part.get_payload()
        candidate = subparts[0] if subparts else None
    if candidate is not None:
        yield from self._find_body(candidate, preferencelist)