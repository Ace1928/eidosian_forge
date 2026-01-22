from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import HAS_LIBCLOUD, DimensionDataModule
from ansible.module_utils.common.text.converters import to_native

        Create a new Dimension Data network module.
        