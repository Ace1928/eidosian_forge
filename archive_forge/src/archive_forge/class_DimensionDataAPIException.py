from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataAPIException(LibcloudError):

    def __init__(self, code, msg, driver):
        self.code = code
        self.msg = msg
        self.driver = driver

    def __str__(self):
        return '{}: {}'.format(self.code, self.msg)

    def __repr__(self):
        return "<DimensionDataAPIException: code='{}', msg='{}'>".format(self.code, self.msg)