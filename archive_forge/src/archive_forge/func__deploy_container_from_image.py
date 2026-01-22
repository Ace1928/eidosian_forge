import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def _deploy_container_from_image(self, name, image, parameters, cont_params, timeout=default_time_out):
    """
        Deploy a new container from the given image

        :param name: the name of the container
        :param image: .ContainerImage

        :param parameters: dictionary describing the source attribute
        :type  parameters ``dict``

        :param cont_params: dictionary describing the container configuration
        :type  cont_params: dict

        :param timeout: Time to wait for the operation before timeout
        :type  timeout: int

        :rtype: :class: .Container
        """
    if cont_params is None:
        raise LXDAPIException(message='cont_params must be a valid dict')
    data = {'name': name, 'source': {'type': 'none'}}
    if parameters:
        data['source'].update(parameters['source'])
    if data['source']['type'] not in LXD_API_IMAGE_SOURCE_TYPE:
        msg = 'source type must in ' + str(LXD_API_IMAGE_SOURCE_TYPE)
        raise LXDAPIException(message=msg)
    data.update(cont_params)
    data = json.dumps(data)
    response = self.connection.request('/%s/containers' % self.version, method='POST', data=data)
    response_dict = response.parse_body()
    assert_response(response_dict=response_dict, status_code=100)
    if not timeout:
        timeout = LXDContainerDriver.default_time_out
    try:
        id = response_dict['metadata']['id']
        req_str = '/{}/operations/{}/wait?timeout={}'.format(self.version, id, timeout)
        response = self.connection.request(req_str)
    except BaseHTTPError as err:
        lxd_exception = self._get_lxd_api_exception_for_error(err)
        if lxd_exception.message != 'not found':
            raise lxd_exception
    return self.get_container(id=name)