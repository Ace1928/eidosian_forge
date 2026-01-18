import os
import base64
from libcloud.utils.py3 import b
from libcloud.common.kubernetes import (

    Test class mixin which tests different type of Kubernetes authentication
    mechanisms (client cert, token, basic auth).

    It's to be used with all the drivers which inherit from KubernetesDriverMixin.
    