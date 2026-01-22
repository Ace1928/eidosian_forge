from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def apply_apidoc_patches(self):
    """
        Apply patches to the local apidoc representation.
        When adding another patch, consider that the endpoint in question may depend on a plugin to be available.
        If possible, make the patch only execute on specific server/plugin versions.
        """
    self._patch_host_update()
    self._patch_subnet_rex_api()
    self._patch_subnet_externalipam_group_api()
    self._patch_organization_update_api()
    self._patch_cv_filter_rule_api()
    self._patch_ak_product_content_per_page()
    self._patch_organization_ignore_types_api()
    self._patch_products_repositories_allow_nil_credential()