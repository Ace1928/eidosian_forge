import glob
import http.client
import os
import re
import tempfile
import time
import traceback
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick.privileged import lightos as priv_lightos
from os_brick import utils
Disconnect a volume from the local host.

        The connection_properties are the same as from connect_volume.
        The device_info is returned from connect_volume.
        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :param device_info: historical difference, but same as connection_props
        :type device_info: dict
        :param force: Whether to forcefully disconnect even if flush fails.
        :type force: bool
        :param ignore_errors: When force is True, this will decide whether to
                              ignore errors or raise an exception once finished
                              the operation.  Default is False.
        :type ignore_errors: bool
        