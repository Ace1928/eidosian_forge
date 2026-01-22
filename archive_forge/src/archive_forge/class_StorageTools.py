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
class StorageTools:
    NAME_CONFIG = 'config.json'

    @staticmethod
    def constructPath(*path):
        path = os.path.join(*path)
        fullPath = os.path.join(user_config_dir(YowConstants.YOWSUP), path)
        if not os.path.exists(os.path.dirname(fullPath)):
            os.makedirs(os.path.dirname(fullPath))
        return fullPath

    @staticmethod
    def getStorageForProfile(profile_name):
        if type(profile_name) is not str:
            profile_name = str(profile_name)
        return StorageTools.constructPath(profile_name)

    @staticmethod
    def writeProfileData(profile_name, name, val):
        logger.debug('writeProfileData(profile_name=%s, name=%s, val=[omitted])' % (profile_name, name))
        path = os.path.join(StorageTools.getStorageForProfile(profile_name), name)
        logger.debug('Writing %s' % path)
        with open(path, 'w' if type(val) is str else 'wb') as attrFile:
            attrFile.write(val)

    @staticmethod
    def readProfileData(profile_name, name, default=None):
        logger.debug('readProfileData(profile_name=%s, name=%s)' % (profile_name, name))
        path = StorageTools.getStorageForProfile(profile_name)
        dataFilePath = os.path.join(path, name)
        if os.path.isfile(dataFilePath):
            logger.debug('Reading %s' % dataFilePath)
            with open(dataFilePath, 'rb') as attrFile:
                return attrFile.read()
        else:
            logger.debug('%s does not exist' % dataFilePath)
        return default

    @classmethod
    def writeProfileConfig(cls, profile_name, config):
        cls.writeProfileData(profile_name, cls.NAME_CONFIG, config)

    @classmethod
    def readProfileConfig(cls, profile_name, config):
        return cls.readProfileData(profile_name, cls.NAME_CONFIG)