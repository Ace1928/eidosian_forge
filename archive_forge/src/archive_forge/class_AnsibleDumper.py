from __future__ import (absolute_import, division, print_function)
import yaml
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.yaml import SafeDumper
from ansible.parsing.yaml.objects import AnsibleUnicode, AnsibleSequence, AnsibleMapping, AnsibleVaultEncryptedUnicode
from ansible.utils.unsafe_proxy import AnsibleUnsafeText, AnsibleUnsafeBytes, NativeJinjaUnsafeText, NativeJinjaText, _is_unsafe
from ansible.template import AnsibleUndefined
from ansible.vars.hostvars import HostVars, HostVarsVars
from ansible.vars.manager import VarsWithSources
class AnsibleDumper(SafeDumper):
    """
    A simple stub class that allows us to add representers
    for our overridden object types.
    """