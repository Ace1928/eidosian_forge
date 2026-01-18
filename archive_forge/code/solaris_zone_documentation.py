from __future__ import absolute_import, division, print_function
import os
import platform
import re
import tempfile
import time
from ansible.module_utils.basic import AnsibleModule

            The boot command can return before the zone has fully booted. This is especially
            true on the first boot when the zone initializes the SMF services. Unless the zone
            has fully booted, subsequent tasks in the playbook may fail as services aren't running yet.
            Wait until the zone's console login is running; once that's running, consider the zone booted.
            