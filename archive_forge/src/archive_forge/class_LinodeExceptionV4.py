from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeExceptionV4(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return '%s' % self.message

    def __repr__(self):
        return "<LinodeException '%s'>" % self.message