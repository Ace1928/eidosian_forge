import binascii
import warnings
from webob.compat import (
def _hide_passwd(items):
    for k, v in items:
        if 'password' in k or 'passwd' in k or 'pwd' in k:
            yield (k, '******')
        else:
            yield (k, v)