from __future__ import absolute_import, division, print_function
import errno
import glob
import json
import os
import re
import re
import re
def _call_action(self, action, params=None, headers=None, data=None, files=None):
    if params is None:
        params = {}
    route = action.find_route(params)
    get_params = {key: value for key, value in params.items() if key not in route.params_in_path}
    return self.http_call(route.method, route.path_with_params(params), get_params, headers, data, files)