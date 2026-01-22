from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
class DockerAPIEngine(Engine):

    def __init__(self, get_value, preprocess_value=None, get_expected_values=None, ignore_mismatching_result=None, set_value=None, update_value=None, can_set_value=None, can_update_value=None, min_api_version=None, compare_value=None, needs_container_image=None, needs_host_info=None, extra_option_minimal_versions=None):
        self.min_api_version = min_api_version
        self.min_api_version_obj = None if min_api_version is None else LooseVersion(min_api_version)
        self.get_value = get_value
        self.set_value = set_value
        self.get_expected_values = get_expected_values or (lambda module, client, api_version, options, image, values, host_info: values)
        self.ignore_mismatching_result = ignore_mismatching_result or (lambda module, client, api_version, option, image, container_value, expected_value: False)
        self.preprocess_value = preprocess_value or (lambda module, client, api_version, options, values: values)
        self.update_value = update_value
        self.can_set_value = can_set_value or (lambda api_version: set_value is not None)
        self.can_update_value = can_update_value or (lambda api_version: update_value is not None)
        self.needs_container_image = needs_container_image or (lambda values: False)
        self.needs_host_info = needs_host_info or (lambda values: False)
        if compare_value is not None:
            self.compare_value = compare_value
        self.extra_option_minimal_versions = extra_option_minimal_versions

    @classmethod
    def config_value(cls, config_name, postprocess_for_get=None, preprocess_for_set=None, get_expected_value=None, ignore_mismatching_result=None, min_api_version=None, preprocess_value=None, update_parameter=None):

        def preprocess_value_(module, client, api_version, options, values):
            if len(options) != 1:
                raise AssertionError('config_value can only be used for a single option')
            if preprocess_value is not None and options[0].name in values:
                value = preprocess_value(module, client, api_version, values[options[0].name])
                if value is None:
                    del values[options[0].name]
                else:
                    values[options[0].name] = value
            return values

        def get_value(module, container, api_version, options, image, host_info):
            if len(options) != 1:
                raise AssertionError('config_value can only be used for a single option')
            value = container['Config'].get(config_name, _SENTRY)
            if postprocess_for_get:
                value = postprocess_for_get(module, api_version, value, _SENTRY)
            if value is _SENTRY:
                return {}
            return {options[0].name: value}
        get_expected_values_ = None
        if get_expected_value:

            def get_expected_values_(module, client, api_version, options, image, values, host_info):
                if len(options) != 1:
                    raise AssertionError('host_config_value can only be used for a single option')
                value = values.get(options[0].name, _SENTRY)
                value = get_expected_value(module, client, api_version, image, value, _SENTRY)
                if value is _SENTRY:
                    return values
                return {options[0].name: value}

        def set_value(module, data, api_version, options, values):
            if len(options) != 1:
                raise AssertionError('config_value can only be used for a single option')
            if options[0].name not in values:
                return
            value = values[options[0].name]
            if preprocess_for_set:
                value = preprocess_for_set(module, api_version, value)
            data[config_name] = value
        update_value = None
        if update_parameter:

            def update_value(module, data, api_version, options, values):
                if len(options) != 1:
                    raise AssertionError('update_parameter can only be used for a single option')
                if options[0].name not in values:
                    return
                value = values[options[0].name]
                if preprocess_for_set:
                    value = preprocess_for_set(module, api_version, value)
                data[update_parameter] = value
        return cls(get_value=get_value, preprocess_value=preprocess_value_, get_expected_values=get_expected_values_, ignore_mismatching_result=ignore_mismatching_result, set_value=set_value, min_api_version=min_api_version, update_value=update_value)

    @classmethod
    def host_config_value(cls, host_config_name, postprocess_for_get=None, preprocess_for_set=None, get_expected_value=None, ignore_mismatching_result=None, min_api_version=None, preprocess_value=None, update_parameter=None):

        def preprocess_value_(module, client, api_version, options, values):
            if len(options) != 1:
                raise AssertionError('host_config_value can only be used for a single option')
            if preprocess_value is not None and options[0].name in values:
                value = preprocess_value(module, client, api_version, values[options[0].name])
                if value is None:
                    del values[options[0].name]
                else:
                    values[options[0].name] = value
            return values

        def get_value(module, container, api_version, options, get_value, host_info):
            if len(options) != 1:
                raise AssertionError('host_config_value can only be used for a single option')
            value = container['HostConfig'].get(host_config_name, _SENTRY)
            if postprocess_for_get:
                value = postprocess_for_get(module, api_version, value, _SENTRY)
            if value is _SENTRY:
                return {}
            return {options[0].name: value}
        get_expected_values_ = None
        if get_expected_value:

            def get_expected_values_(module, client, api_version, options, image, values, host_info):
                if len(options) != 1:
                    raise AssertionError('host_config_value can only be used for a single option')
                value = values.get(options[0].name, _SENTRY)
                value = get_expected_value(module, client, api_version, image, value, _SENTRY)
                if value is _SENTRY:
                    return values
                return {options[0].name: value}

        def set_value(module, data, api_version, options, values):
            if len(options) != 1:
                raise AssertionError('host_config_value can only be used for a single option')
            if options[0].name not in values:
                return
            if 'HostConfig' not in data:
                data['HostConfig'] = {}
            value = values[options[0].name]
            if preprocess_for_set:
                value = preprocess_for_set(module, api_version, value)
            data['HostConfig'][host_config_name] = value
        update_value = None
        if update_parameter:

            def update_value(module, data, api_version, options, values):
                if len(options) != 1:
                    raise AssertionError('update_parameter can only be used for a single option')
                if options[0].name not in values:
                    return
                value = values[options[0].name]
                if preprocess_for_set:
                    value = preprocess_for_set(module, api_version, value)
                data[update_parameter] = value
        return cls(get_value=get_value, preprocess_value=preprocess_value_, get_expected_values=get_expected_values_, ignore_mismatching_result=ignore_mismatching_result, set_value=set_value, min_api_version=min_api_version, update_value=update_value)