import base64
import errno
import hashlib
import logging
import zlib
from debtcollector import removals
from keystoneclient import exceptions
from keystoneclient.i18n import _
class OpensslCmsExitStatus(object):
    SUCCESS = 0
    COMMAND_OPTIONS_PARSING_ERROR = 1
    INPUT_FILE_READ_ERROR = 2
    CREATE_CMS_READ_MIME_ERROR = 3