from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def update_svm_name(self, api_root, rest_api, headers, svm_name):
    we, err = self.get_working_environment_property(rest_api, headers, ['ontapClusterProperties.fields(upgradeVersions)'])
    if err is not None:
        return (False, 'Error: get_working_environment_property failed: %s' % str(err))
    body = {'svmNewName': svm_name, 'svmName': we['svmName']}
    response, err, dummy = rest_api.put(api_root + 'svm', body, header=headers)
    if err is not None:
        return (False, 'update svm_name error')
    return (True, None)