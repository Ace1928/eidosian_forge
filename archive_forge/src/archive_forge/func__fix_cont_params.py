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
@staticmethod
def _fix_cont_params(architecture, profiles, ephemeral, config, devices, instance_type):
    """
        Returns a dict with the container parameters
        """
    cont_params = {}
    if architecture is not None:
        cont_params['architecture'] = architecture
    if profiles is not None:
        cont_params['profiles'] = profiles
    else:
        cont_params['profiles'] = [LXDContainerDriver.default_profiles]
    if ephemeral is not None:
        cont_params['ephemeral'] = ephemeral
    else:
        cont_params['ephemeral'] = LXDContainerDriver.default_ephemeral
    if config is not None:
        cont_params['config'] = config
    if devices is not None:
        cont_params['devices'] = devices
    if instance_type is not None:
        cont_params['instance_type'] = instance_type
    return cont_params