from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _run_command_with_sudo_password(self, cmd):
    rc, out, err = ('', '', '')
    with tempfile.NamedTemporaryFile() as sudo_askpass_file:
        sudo_askpass_file.write(b"#!/bin/sh\n\necho '%s'\n" % to_bytes(self.sudo_password))
        os.chmod(sudo_askpass_file.name, 448)
        sudo_askpass_file.file.close()
        rc, out, err = self.module.run_command(cmd, environ_update={'SUDO_ASKPASS': sudo_askpass_file.name})
        self.module.add_cleanup_file(sudo_askpass_file.name)
    return (rc, out, err)