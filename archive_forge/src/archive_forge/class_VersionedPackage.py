import sys
import pytest
from requests.help import info
class VersionedPackage(object):

    def __init__(self, version):
        self.__version__ = version