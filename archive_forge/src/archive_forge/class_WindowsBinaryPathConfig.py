from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
class WindowsBinaryPathConfig(object):

    def __init__(self, ecp, ecp_client, tls_offload):
        self.ecp = ecp if ecp else os.path.join(get_bin_folder(), 'ecp.exe')
        self.ecp_client = ecp_client if ecp_client else os.path.join(get_platform_folder(), 'libecp.dll')
        self.tls_offload = tls_offload if tls_offload else os.path.join(get_platform_folder(), 'libtls_offload.dll')