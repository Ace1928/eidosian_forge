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
@staticmethod
def writeProfileData(profile_name, name, val):
    logger.debug('writeProfileData(profile_name=%s, name=%s, val=[omitted])' % (profile_name, name))
    path = os.path.join(StorageTools.getStorageForProfile(profile_name), name)
    logger.debug('Writing %s' % path)
    with open(path, 'w' if type(val) is str else 'wb') as attrFile:
        attrFile.write(val)