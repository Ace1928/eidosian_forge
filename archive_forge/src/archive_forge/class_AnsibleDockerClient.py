from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
class AnsibleDockerClient(AnsibleDockerClientBase):

    def __init__(self, argument_spec=None, supports_check_mode=False, mutually_exclusive=None, required_together=None, required_if=None, required_one_of=None, required_by=None, min_docker_version=None, min_docker_api_version=None, option_minimal_versions=None, option_minimal_versions_ignore_params=None, fail_results=None):
        self.fail_results = fail_results or {}
        merged_arg_spec = dict()
        merged_arg_spec.update(DOCKER_COMMON_ARGS)
        if argument_spec:
            merged_arg_spec.update(argument_spec)
            self.arg_spec = merged_arg_spec
        mutually_exclusive_params = []
        mutually_exclusive_params += DOCKER_MUTUALLY_EXCLUSIVE
        if mutually_exclusive:
            mutually_exclusive_params += mutually_exclusive
        required_together_params = []
        required_together_params += DOCKER_REQUIRED_TOGETHER
        if required_together:
            required_together_params += required_together
        self.module = AnsibleModule(argument_spec=merged_arg_spec, supports_check_mode=supports_check_mode, mutually_exclusive=mutually_exclusive_params, required_together=required_together_params, required_if=required_if, required_one_of=required_one_of, required_by=required_by or {})
        self.debug = self.module.params.get('debug')
        self.check_mode = self.module.check_mode
        super(AnsibleDockerClient, self).__init__(min_docker_version=min_docker_version, min_docker_api_version=min_docker_api_version)
        if option_minimal_versions is not None:
            self._get_minimal_versions(option_minimal_versions, option_minimal_versions_ignore_params)

    def fail(self, msg, **kwargs):
        self.fail_results.update(kwargs)
        self.module.fail_json(msg=msg, **sanitize_result(self.fail_results))

    def deprecate(self, msg, version=None, date=None, collection_name=None):
        self.module.deprecate(msg, version=version, date=date, collection_name=collection_name)

    def _get_params(self):
        return self.module.params

    def _get_minimal_versions(self, option_minimal_versions, ignore_params=None):
        self.option_minimal_versions = dict()
        for option in self.module.argument_spec:
            if ignore_params is not None:
                if option in ignore_params:
                    continue
            self.option_minimal_versions[option] = dict()
        self.option_minimal_versions.update(option_minimal_versions)
        for option, data in self.option_minimal_versions.items():
            support_docker_py = True
            support_docker_api = True
            if 'docker_py_version' in data:
                support_docker_py = self.docker_py_version >= LooseVersion(data['docker_py_version'])
            if 'docker_api_version' in data:
                support_docker_api = self.docker_api_version >= LooseVersion(data['docker_api_version'])
            data['supported'] = support_docker_py and support_docker_api
            if not data['supported']:
                if 'detect_usage' in data:
                    used = data['detect_usage'](self)
                else:
                    used = self.module.params.get(option) is not None
                    if used and 'default' in self.module.argument_spec[option]:
                        used = self.module.params[option] != self.module.argument_spec[option]['default']
                if used:
                    if 'usage_msg' in data:
                        usg = data['usage_msg']
                    else:
                        usg = 'set %s option' % (option,)
                    if not support_docker_api:
                        msg = 'Docker API version is %s. Minimum version required is %s to %s.'
                        msg = msg % (self.docker_api_version_str, data['docker_api_version'], usg)
                    elif not support_docker_py:
                        msg = "Docker SDK for Python version is %s (%s's Python %s). Minimum version required is %s to %s. "
                        if LooseVersion(data['docker_py_version']) < LooseVersion('2.0.0'):
                            msg += DOCKERPYUPGRADE_RECOMMEND_DOCKER
                        elif self.docker_py_version < LooseVersion('2.0.0'):
                            msg += DOCKERPYUPGRADE_SWITCH_TO_DOCKER
                        else:
                            msg += DOCKERPYUPGRADE_UPGRADE_DOCKER
                        msg = msg % (docker_version, platform.node(), sys.executable, data['docker_py_version'], usg)
                    else:
                        msg = 'Cannot %s with your configuration.' % (usg,)
                    self.fail(msg)

    def report_warnings(self, result, warnings_key=None):
        """
        Checks result of client operation for warnings, and if present, outputs them.

        warnings_key should be a list of keys used to crawl the result dictionary.
        For example, if warnings_key == ['a', 'b'], the function will consider
        result['a']['b'] if these keys exist. If the result is a non-empty string, it
        will be reported as a warning. If the result is a list, every entry will be
        reported as a warning.

        In most cases (if warnings are returned at all), warnings_key should be
        ['Warnings'] or ['Warning']. The default value (if not specified) is ['Warnings'].
        """
        if warnings_key is None:
            warnings_key = ['Warnings']
        for key in warnings_key:
            if not isinstance(result, Mapping):
                return
            result = result.get(key)
        if isinstance(result, Sequence):
            for warning in result:
                self.module.warn('Docker warning: {0}'.format(warning))
        elif isinstance(result, string_types) and result:
            self.module.warn('Docker warning: {0}'.format(result))