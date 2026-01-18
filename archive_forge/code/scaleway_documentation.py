import copy
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider

        Delete an existing key pair.

        :param key_pair: Key pair object.
        :type key_pair: :class:`KeyPair`

        :return:   True of False based on success of Keypair deletion
        :rtype:    ``bool``
        