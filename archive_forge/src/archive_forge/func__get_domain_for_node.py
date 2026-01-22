import os
import re
import time
import platform
import mimetypes
import subprocess
from os.path import join as pjoin
from collections import defaultdict
from libcloud.utils.py3 import ET, ensure_string
from libcloud.compute.base import Node, NodeState, NodeDriver
from libcloud.compute.types import Provider
from libcloud.utils.networking import is_public_subnet
def _get_domain_for_node(self, node):
    """
        Return libvirt domain object for the provided node.
        """
    domain = self.connection.lookupByUUIDString(node.uuid)
    return domain