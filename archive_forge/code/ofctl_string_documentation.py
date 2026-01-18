import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext

    Convert an ovs-ofctl style NXM_/OXM_ field name to
    a os_ken match field name.
    