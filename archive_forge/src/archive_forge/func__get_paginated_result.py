from __future__ import absolute_import, division, print_function
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
def _get_paginated_result(self, operation, **options):
    """return all results of a paginated api response"""
    records_pagination = operation(per_page=self.pagination_per_page, **options).pagination
    result_list = []
    for page in range(1, records_pagination.total_pages + 1):
        page_data = operation(per_page=self.pagination_per_page, page=page, **options).data
        result_list.extend(page_data)
    return result_list