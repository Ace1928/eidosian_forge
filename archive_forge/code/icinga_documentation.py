from __future__ import absolute_import, division, print_function
import json
from collections import defaultdict
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import iteritems
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six.moves.urllib.parse import quote as urlquote

        Create, update or delete the objects in the director.

        Parameters:
            state: type str, the state of the object, present or absent

        Returns:
            changed: whether the object was changed
            diff_result: the diff of the object
        