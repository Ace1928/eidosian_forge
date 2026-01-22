from __future__ import (annotations, absolute_import, division, print_function)
import base64
import logging
import os
import re
import traceback
import json
import tempfile
import shlex
import subprocess
import time
import typing as t
import xml.etree.ElementTree as ET
from inspect import getfullargspec
from urllib.parse import urlunsplit
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.errors import AnsibleFileNotFound
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.plugins.shell.powershell import _parse_clixml
from ansible.plugins.shell.powershell import ShellBase as PowerShellBase
from ansible.utils.hashing import secure_hash
from ansible.utils.display import Display
def _build_winrm_kwargs(self) -> None:
    self._winrm_host = self.get_option('remote_addr')
    self._winrm_user = self.get_option('remote_user')
    self._winrm_pass = self.get_option('remote_password')
    self._winrm_port = self.get_option('port')
    self._winrm_scheme = self.get_option('scheme')
    if self._winrm_scheme is None:
        self._winrm_scheme = 'http' if self._winrm_port == 5985 else 'https'
    self._winrm_path = self.get_option('path')
    self._kinit_cmd = self.get_option('kerberos_command')
    self._winrm_transport = self.get_option('transport')
    self._winrm_connection_timeout = self.get_option('connection_timeout')
    if hasattr(winrm, 'FEATURE_SUPPORTED_AUTHTYPES'):
        self._winrm_supported_authtypes = set(winrm.FEATURE_SUPPORTED_AUTHTYPES)
    else:
        self._winrm_supported_authtypes = set(['plaintext', 'ssl', 'kerberos'])
    if self._winrm_transport is None or self._winrm_transport[0] is None:
        transport_selector = ['ssl'] if self._winrm_scheme == 'https' else ['plaintext']
        if HAVE_KERBEROS and (self._winrm_user and '@' in self._winrm_user):
            self._winrm_transport = ['kerberos'] + transport_selector
        else:
            self._winrm_transport = transport_selector
    unsupported_transports = set(self._winrm_transport).difference(self._winrm_supported_authtypes)
    if unsupported_transports:
        raise AnsibleError('The installed version of WinRM does not support transport(s) %s' % to_native(list(unsupported_transports), nonstring='simplerepr'))
    kinit_mode = self.get_option('kerberos_mode')
    if kinit_mode is None:
        self._kerb_managed = 'kerberos' in self._winrm_transport and (self._winrm_pass is not None and self._winrm_pass != '')
    elif kinit_mode == 'managed':
        self._kerb_managed = True
    elif kinit_mode == 'manual':
        self._kerb_managed = False
    internal_kwarg_mask = {'self', 'endpoint', 'transport', 'username', 'password', 'scheme', 'path', 'kinit_mode', 'kinit_cmd'}
    self._winrm_kwargs = dict(username=self._winrm_user, password=self._winrm_pass)
    argspec = getfullargspec(Protocol.__init__)
    supported_winrm_args = set(argspec.args)
    supported_winrm_args.update(internal_kwarg_mask)
    passed_winrm_args = {v.replace('ansible_winrm_', '') for v in self.get_option('_extras')}
    unsupported_args = passed_winrm_args.difference(supported_winrm_args)
    for arg in unsupported_args:
        display.warning('ansible_winrm_{0} unsupported by pywinrm (is an up-to-date version of pywinrm installed?)'.format(arg))
    for arg in passed_winrm_args.difference(internal_kwarg_mask).intersection(supported_winrm_args):
        self._winrm_kwargs[arg] = self.get_option('_extras')['ansible_winrm_%s' % arg]