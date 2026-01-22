import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def _prepare_set(msg, maintype, subtype, headers):
    msg['Content-Type'] = '/'.join((maintype, subtype))
    if headers:
        if not hasattr(headers[0], 'name'):
            mp = msg.policy
            headers = [mp.header_factory(*mp.header_source_parse([header])) for header in headers]
        try:
            for header in headers:
                if header.defects:
                    raise header.defects[0]
                msg[header.name] = header
        except email.errors.HeaderDefect as exc:
            raise ValueError('Invalid header: {}'.format(header.fold(policy=msg.policy))) from exc