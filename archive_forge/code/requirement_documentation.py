from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.playbook.role.definition import RoleDefinition
from ansible.utils.display import Display
from ansible.utils.galaxy import scm_archive_resource

    Helper class for Galaxy, which is used to parse both dependencies
    specified in meta/main.yml and requirements.yml files.
    