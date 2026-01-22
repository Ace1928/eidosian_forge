from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
@six.add_metaclass(abc.ABCMeta)
class PSAdapter(object):
    NAME_ATTRS = ('name', 'cmdline')
    PATTERN_ATTRS = ('name', 'exe', 'cmdline')

    def __init__(self, psutil):
        self._psutil = psutil

    @staticmethod
    def from_package(psutil):
        version = LooseVersion(psutil.__version__)
        if version < LooseVersion('2.0.0'):
            return PSAdapter100(psutil)
        elif version < LooseVersion('5.3.0'):
            return PSAdapter200(psutil)
        else:
            return PSAdapter530(psutil)

    def get_pids_by_name(self, name):
        return [p.pid for p in self._process_iter(*self.NAME_ATTRS) if self._has_name(p, name)]

    def _process_iter(self, *attrs):
        return self._psutil.process_iter()

    def _has_name(self, proc, name):
        attributes = self._get_proc_attributes(proc, *self.NAME_ATTRS)
        return compare_lower(attributes['name'], name) or (attributes['cmdline'] and compare_lower(attributes['cmdline'][0], name))

    def _get_proc_attributes(self, proc, *attributes):
        return dict(((attribute, self._get_attribute_from_proc(proc, attribute)) for attribute in attributes))

    @staticmethod
    @abc.abstractmethod
    def _get_attribute_from_proc(proc, attribute):
        pass

    def get_pids_by_pattern(self, pattern, ignore_case):
        flags = 0
        if ignore_case:
            flags |= re.I
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise PSAdapterError("'%s' is not a valid regular expression: %s" % (pattern, to_native(e)))
        return [p.pid for p in self._process_iter(*self.PATTERN_ATTRS) if self._matches_regex(p, regex)]

    def _matches_regex(self, proc, regex):
        attributes = self._get_proc_attributes(proc, *self.PATTERN_ATTRS)
        matches_name = regex.search(to_native(attributes['name']))
        matches_exe = attributes['exe'] and regex.search(basename(to_native(attributes['exe'])))
        matches_cmd = attributes['cmdline'] and regex.search(to_native(' '.join(attributes['cmdline'])))
        return any([matches_name, matches_exe, matches_cmd])