from __future__ import absolute_import, division, print_function
import abc
import binascii
import os
from base64 import b64encode
from datetime import datetime
from hashlib import sha256
from ansible.module_utils import six
from ansible.module_utils.common.text.converters import to_text
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import convert_relative_to_datetime
from ansible_collections.community.crypto.plugins.module_utils.openssh.utils import (
def apply_directives(directives):
    if any((d not in _DIRECTIVES for d in directives)):
        raise ValueError('directives must be one of %s' % ', '.join(_DIRECTIVES))
    directive_to_option = {'no-x11-forwarding': OpensshCertificateOption('extension', 'permit-x11-forwarding', ''), 'no-agent-forwarding': OpensshCertificateOption('extension', 'permit-agent-forwarding', ''), 'no-port-forwarding': OpensshCertificateOption('extension', 'permit-port-forwarding', ''), 'no-pty': OpensshCertificateOption('extension', 'permit-pty', ''), 'no-user-rc': OpensshCertificateOption('extension', 'permit-user-rc', '')}
    if 'clear' in directives:
        return []
    else:
        return list(set(default_options()) - set((directive_to_option[d] for d in directives)))