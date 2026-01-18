from __future__ import annotations
from typing import Any, Callable  # noqa: H301
from oslo_log import log as logging
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.remotefs import remotefs
from os_brick import utils
No need to do anything to disconnect a volume in a filesystem.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :param device_info: historical difference, but same as connection_props
        :type device_info: dict
        