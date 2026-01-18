from __future__ import absolute_import, division, print_function
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def test_version_requirement(opt):
    req_version = helm_pull_opt_versionning.get(opt)
    if req_version and LooseVersion(helm_version) < LooseVersion(req_version):
        module.fail_json(msg='Parameter {0} requires helm >= {1}, current version is {2}'.format(opt, req_version, helm_version))