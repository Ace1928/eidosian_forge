from __future__ import absolute_import, division, print_function
import os
import re
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule, create_multipart_formdata, request
from ansible.module_utils._text import to_native
Apply firmware policy has been enforced on E-Series storage system.