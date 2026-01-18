from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
 Cast or dispel spells.

    This manages the whole system ('*'), list or a single spell. Command 'cast'
    is used to install or rebuild spells, while 'dispel' takes care of theirs
    removal from the system.

    