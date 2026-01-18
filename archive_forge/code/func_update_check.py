from __future__ import (absolute_import, division, print_function)
import inspect
import os
import time
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ansible_collections.ovirt.ovirt.plugins.module_utils.cloud import CloudRetry
from ansible_collections.ovirt.ovirt.plugins.module_utils.version import ComparableVersion
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common._collections_compat import Mapping
def update_check(self, entity):
    """
        This method handle checks whether the entity values are same as values
        passed to ansible module. By default we don't compare any values.

        :param entity: Entity we want to compare with Ansible module values.
        :return: True if values are same, so we don't need to update the entity.
        """
    return True