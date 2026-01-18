import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
def token_to_cms(signed_text):
    """Convert a custom formatted token to a PEM-formatted token.

    See documentation for cms_to_token() for details on the custom formatting.
    """
    copy_of_text = signed_text.replace('-', '/')
    lines = ['-----BEGIN CMS-----']
    lines += [copy_of_text[n:n + 64] for n in range(0, len(copy_of_text), 64)]
    lines.append('-----END CMS-----\n')
    return '\n'.join(lines)