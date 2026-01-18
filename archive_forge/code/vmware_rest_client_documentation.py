from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native

        Return tag object by category name
        Args:
            tag_name: Name of tag
            category_id: Id of category
        Returns: Tag object if found else None
        