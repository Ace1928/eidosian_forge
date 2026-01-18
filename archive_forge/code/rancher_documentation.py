import base64
from libcloud.utils.py3 import b, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.container.providers import Provider

        Convert container in proper Container instance object
        ** Updating is NOT supported!!

        :param data: API data about container i.e. result.object
        :return: Proper Container object:
         see http://libcloud.readthedocs.io/en/latest/container/api.html

        