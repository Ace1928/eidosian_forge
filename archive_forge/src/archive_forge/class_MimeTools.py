import os
from .constants import YowConstants
import codecs, sys
import logging
import tempfile
import base64
import hashlib
import os.path, mimetypes
import uuid
from consonance.structs.keypair import KeyPair
from appdirs import user_config_dir
from .optionalmodules import PILOptionalModule, FFVideoOptionalModule
class MimeTools:
    MIME_FILE = os.path.join(os.path.dirname(__file__), 'mime.types')
    mimetypes.init()
    try:
        mimetypes.init([MIME_FILE])
    except exception as e:
        logger.warning("Mime types supported can't be read. System mimes will be used. Cause: " + e.message)

    @staticmethod
    def getMIME(filepath):
        mimeType = mimetypes.guess_type(filepath)[0]
        if mimeType is None:
            raise Exception('Unsupported/unrecognized file type for: ' + filepath)
        return mimeType