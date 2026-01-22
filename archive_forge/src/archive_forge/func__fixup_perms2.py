from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import random
import re
import shlex
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleActionSkip, AnsibleActionFail, AnsibleAuthenticationFailure
from ansible.executor.module_common import modify_module
from ansible.executor.interpreter_discovery import discover_interpreter, InterpreterDiscoveryRequiredError
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.module_utils.errors import UnsupportedError
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.jsonify import jsonify
from ansible.release import __version__
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var, AnsibleUnsafeText
from ansible.vars.clean import remove_internal_keys
from ansible.utils.plugin_docs import get_versioned_doclink
def _fixup_perms2(self, remote_paths, remote_user=None, execute=True):
    """
        We need the files we upload to be readable (and sometimes executable)
        by the user being sudo'd to but we want to limit other people's access
        (because the files could contain passwords or other private
        information.  We achieve this in one of these ways:

        * If no sudo is performed or the remote_user is sudo'ing to
          themselves, we don't have to change permissions.
        * If the remote_user sudo's to a privileged user (for instance, root),
          we don't have to change permissions
        * If the remote_user sudo's to an unprivileged user then we attempt to
          grant the unprivileged user access via file system acls.
        * If granting file system acls fails we try to change the owner of the
          file with chown which only works in case the remote_user is
          privileged or the remote systems allows chown calls by unprivileged
          users (e.g. HP-UX)
        * If the above fails, we next try 'chmod +a' which is a macOS way of
          setting ACLs on files.
        * If the above fails, we check if ansible_common_remote_group is set.
          If it is, we attempt to chgrp the file to its value. This is useful
          if the remote_user has a group in common with the become_user. As the
          remote_user, we can chgrp the file to that group and allow the
          become_user to read it.
        * If (the chown fails AND ansible_common_remote_group is not set) OR
          (ansible_common_remote_group is set AND the chgrp (or following chmod)
          returned non-zero), we can set the file to be world readable so that
          the second unprivileged user can read the file.
          Since this could allow other users to get access to private
          information we only do this if ansible is configured with
          "allow_world_readable_tmpfiles" in the ansible.cfg. Also note that
          when ansible_common_remote_group is set this final fallback is very
          unlikely to ever be triggered, so long as chgrp was successful. But
          just because the chgrp was successful, does not mean Ansible can
          necessarily access the files (if, for example, the variable was set
          to a group that remote_user is in, and can chgrp to, but does not have
          in common with become_user).
        """
    if remote_user is None:
        remote_user = self._get_remote_user()
    if getattr(self._connection._shell, '_IS_WINDOWS', False):
        return remote_paths
    if not self._is_become_unprivileged():
        if execute:
            res = self._remote_chmod(remote_paths, 'u+x')
            if res['rc'] != 0:
                raise AnsibleError('Failed to set execute bit on remote files (rc: {0}, err: {1})'.format(res['rc'], to_native(res['stderr'])))
        return remote_paths
    become_user = self.get_become_option('become_user')
    if execute:
        chmod_mode = 'rx'
        setfacl_mode = 'r-x'
        chmod_acl_mode = '{0} allow read,execute'.format(become_user)
        posix_acl_mode = 'A+user:{0}:rx:allow'.format(become_user)
    else:
        chmod_mode = 'rX'
        setfacl_mode = 'r-X'
        chmod_acl_mode = '{0} allow read'.format(become_user)
        posix_acl_mode = 'A+user:{0}:r:allow'.format(become_user)
    res = self._remote_set_user_facl(remote_paths, become_user, setfacl_mode)
    if res['rc'] == 0:
        return remote_paths
    if execute:
        res = self._remote_chmod(remote_paths, 'u+x')
        if res['rc'] != 0:
            raise AnsibleError('Failed to set file mode or acl on remote temporary files (rc: {0}, err: {1})'.format(res['rc'], to_native(res['stderr'])))
    res = self._remote_chown(remote_paths, become_user)
    if res['rc'] == 0:
        return remote_paths
    if remote_user in self._get_admin_users():
        raise AnsibleError('Failed to change ownership of the temporary files Ansible (via chmod nor setfacl) needs to create despite connecting as a privileged user. Unprivileged become user would be unable to read the file.')
    try:
        res = self._remote_chmod([chmod_acl_mode] + list(remote_paths), '+a')
    except AnsibleAuthenticationFailure as e:
        pass
    else:
        if res['rc'] == 0:
            return remote_paths
    res = self._remote_chmod(remote_paths, posix_acl_mode)
    if res['rc'] == 0:
        return remote_paths
    become_link = get_versioned_doclink('playbook_guide/playbooks_privilege_escalation.html')
    group = self.get_shell_option('common_remote_group')
    if group is not None:
        res = self._remote_chgrp(remote_paths, group)
        if res['rc'] == 0:
            if self.get_shell_option('world_readable_temp'):
                display.warning('Both common_remote_group and allow_world_readable_tmpfiles are set. chgrp was successful, but there is no guarantee that Ansible will be able to read the files after this operation, particularly if common_remote_group was set to a group of which the unprivileged become user is not a member. In this situation, allow_world_readable_tmpfiles is a no-op. See this URL for more details: %s#risks-of-becoming-an-unprivileged-user' % become_link)
            if execute:
                group_mode = 'g+rwx'
            else:
                group_mode = 'g+rw'
            res = self._remote_chmod(remote_paths, group_mode)
            if res['rc'] == 0:
                return remote_paths
    if self.get_shell_option('world_readable_temp'):
        display.warning('Using world-readable permissions for temporary files Ansible needs to create when becoming an unprivileged user. This may be insecure. For information on securing this, see %s#risks-of-becoming-an-unprivileged-user' % become_link)
        res = self._remote_chmod(remote_paths, 'a+%s' % chmod_mode)
        if res['rc'] == 0:
            return remote_paths
        raise AnsibleError('Failed to set file mode on remote files (rc: {0}, err: {1})'.format(res['rc'], to_native(res['stderr'])))
    raise AnsibleError('Failed to set permissions on the temporary files Ansible needs to create when becoming an unprivileged user (rc: %s, err: %s}). For information on working around this, see %s#risks-of-becoming-an-unprivileged-user' % (res['rc'], to_native(res['stderr']), become_link))