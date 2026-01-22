from __future__ import absolute_import, division, print_function
import errno
import json
import os
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.image_archive import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import (
from ansible_collections.community.docker.plugins.module_utils._api.constants import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException, NotFound
from ansible_collections.community.docker.plugins.module_utils._api.utils.build import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
@staticmethod
def archived_image_action(failure_logger, archive_path, current_image_name, current_image_id):
    """
        If the archive is missing or requires replacement, return an action message.

        :param failure_logger: a logging function that accepts one parameter of type str
        :type failure_logger: Callable
        :param archive_path: Filename to write archive to
        :type archive_path: str
        :param current_image_name: repo:tag
        :type current_image_name: str
        :param current_image_id: Hash, including hash type prefix such as "sha256:"
        :type current_image_id: str

        :returns: Either None, or an Ansible action message.
        :rtype: str
        """

    def build_msg(reason):
        return 'Archived image %s to %s, %s' % (current_image_name, archive_path, reason)
    try:
        archived = archived_image_manifest(archive_path)
    except ImageArchiveInvalidException as exc:
        failure_logger('Unable to extract manifest summary from archive: %s' % to_native(exc))
        return build_msg('overwriting an unreadable archive file')
    if archived is None:
        return build_msg('since none present')
    elif current_image_id == api_image_id(archived.image_id) and [current_image_name] == archived.repo_tags:
        return None
    else:
        name = ', '.join(archived.repo_tags)
        return build_msg('overwriting archive with image %s named %s' % (archived.image_id, name))