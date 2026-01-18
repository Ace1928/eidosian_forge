from __future__ import annotations
import glob
import os
from typing import Iterable
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick.initiator import linuxscsi
Write the LUN to the port's unit_remove attribute.

        If auto-discovery of LUNs is disabled on s390 platforms
        luns need to be removed from the configuration through the
        unit_remove interface
        