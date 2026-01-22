import base64
import hashlib
import logging
import os
import re
import shutil
import ssl
import struct
import tempfile
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, Password, unify_credentials
from spnego._credssp_structures import (
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.tls import (
def _wrap_ssl_error(context: str) -> typing.Callable[[F], F]:

    def decorator(func: F) -> F:

        def wrapped(*args: typing.Any, **kwargs: typing.Any) -> F:
            try:
                return func(*args, **kwargs)
            except ssl.SSLError as e:
                raise SpnegoError(error_code=ErrorCode.failure, context_msg='%s: %s' % (context, e)) from e
        return typing.cast(F, wrapped)
    return decorator