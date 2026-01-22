import datetime
import sys
import encodings.idna
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url
from urllib3.exceptions import (
from io import UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict
from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import (
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import SSLError as RequestsSSLError
from ._internal_utils import to_native_string, unicode_is_ascii
from .utils import (
from .compat import (
from .compat import json as complexjson
from .status_codes import codes
@staticmethod
def _encode_files(files, data):
    """Build the body for a multipart/form-data request.

        Will successfully encode files when passed as a dict or a list of
        tuples. Order is retained if data is a list of tuples but arbitrary
        if parameters are supplied as a dict.
        The tuples may be 2-tuples (filename, fileobj), 3-tuples (filename, fileobj, contentype)
        or 4-tuples (filename, fileobj, contentype, custom_headers).
        """
    if not files:
        raise ValueError('Files must be provided.')
    elif isinstance(data, basestring):
        raise ValueError('Data must not be a string.')
    new_fields = []
    fields = to_key_val_list(data or {})
    files = to_key_val_list(files or {})
    for field, val in fields:
        if isinstance(val, basestring) or not hasattr(val, '__iter__'):
            val = [val]
        for v in val:
            if v is not None:
                if not isinstance(v, bytes):
                    v = str(v)
                new_fields.append((field.decode('utf-8') if isinstance(field, bytes) else field, v.encode('utf-8') if isinstance(v, str) else v))
    for k, v in files:
        ft = None
        fh = None
        if isinstance(v, (tuple, list)):
            if len(v) == 2:
                fn, fp = v
            elif len(v) == 3:
                fn, fp, ft = v
            else:
                fn, fp, ft, fh = v
        else:
            fn = guess_filename(v) or k
            fp = v
        if isinstance(fp, (str, bytes, bytearray)):
            fdata = fp
        elif hasattr(fp, 'read'):
            fdata = fp.read()
        elif fp is None:
            continue
        else:
            fdata = fp
        rf = RequestField(name=k, data=fdata, filename=fn, headers=fh)
        rf.make_multipart(content_type=ft)
        new_fields.append(rf)
    body, content_type = encode_multipart_formdata(new_fields)
    return (body, content_type)