from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule

    Check the package on AIX.
    It verifies if the package is installed and information

    :param module: Ansible module parameters spec.
    :param package: Package/fileset name.
    :param repository_path: Repository package path.
    :return: Bool, package data.
    