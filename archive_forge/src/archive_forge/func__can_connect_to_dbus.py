from __future__ import absolute_import, division, print_function
from os.path import isfile
from os import getuid, unlink
import re
import shutil
import tempfile
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves import configparser
from ansible.module_utils import distro
def _can_connect_to_dbus(self):
    """
        Checks whether it is possible to connect to the system D-Bus bus.

        :returns: bool -- whether it is possible to connect to the system D-Bus bus.
        """
    try:
        import dbus
    except ImportError:
        self.module.debug('dbus Python module not available, will use CLI')
        return False
    try:
        bus = dbus.SystemBus()
        msg = dbus.lowlevel.SignalMessage('/', 'com.example', 'test')
        bus.send_message(msg)
        bus.flush()
    except dbus.exceptions.DBusException as e:
        self.module.debug('Failed to connect to system D-Bus bus, will use CLI: %s' % e)
        return False
    self.module.debug('Verified system D-Bus bus as usable')
    return True