from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
def _get_potential_volume_paths(self, connection_properties: dict) -> list[str]:
    """Build a list of potential volume paths that exist.

        Given a list of target_portals in the connection_properties,
        a list of paths might exist on the system during discovery.
        This method's job is to build that list of potential paths
        for a volume that might show up.

        This is only used during get_volume_paths time, we are looking to
        find a list of existing volume paths for the connection_properties.
        In this case, we don't want to connect to the portal.  If we
        blindly try and connect to a portal, it could create a new iSCSI
        session that didn't exist previously, and then leave it stale.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :returns: list
        """
    if self.use_multipath:
        LOG.info('Multipath discovery for iSCSI enabled')
        host_devices = self._get_device_path(connection_properties)
    else:
        LOG.info('Multipath discovery for iSCSI not enabled.')
        iscsi_sessions = self._get_iscsi_sessions()
        host_devices_set: set = set()
        for props in self._iterate_all_targets(connection_properties):
            if props['target_portal'] in iscsi_sessions:
                paths = self._get_device_path(props)
                host_devices_set.update(paths)
        host_devices = list(host_devices_set)
    return host_devices