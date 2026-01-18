from __future__ import absolute_import, division, print_function
import binascii
import random
import re
from ansible.module_utils import six
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from time import sleep
Apply state changes to the storage array's truststore.