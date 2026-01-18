from __future__ import absolute_import, division, print_function
import os
from time import sleep
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native, to_text, to_bytes
Upgrade controller firmware.