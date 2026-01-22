import types
from boto.gs.user import User
from boto.exception import InvalidEncryptionConfigError
from xml.sax import handler
Convert EncryptionConfig object into XML string representation.