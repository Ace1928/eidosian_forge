import json
import datetime
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.compute.base import (
from libcloud.compute.types import Provider, NodeState, InvalidCredsError
import asyncio

        List snapshots for a volume.

        :param volume: Volume to list snapshots for
        :type volume: class:`StorageVolume`

        :return: List of volume snapshots.
        :rtype: ``list`` of :class: `VolumeSnapshot`
        