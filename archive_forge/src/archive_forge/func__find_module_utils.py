from __future__ import (absolute_import, division, print_function)
import ast
import base64
import datetime
import json
import os
import shlex
import time
import zipfile
import re
import pkgutil
from ast import AST, Import, ImportFrom
from io import BytesIO
from ansible.release import __version__, __author__
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.executor.interpreter_discovery import InterpreterDiscoveryRequiredError
from ansible.executor.powershell import module_manifest as ps_manifest
from ansible.module_utils.common.json import AnsibleJSONEncoder
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.plugins.loader import module_utils_loader
from ansible.utils.collection_loader._collection_finder import _get_collection_metadata, _nested_dict_get
from ansible.executor import action_write_locks
from ansible.utils.display import Display
from collections import namedtuple
import importlib.util
import importlib.machinery
import sys
import {1} as mod
def _find_module_utils(module_name, b_module_data, module_path, module_args, task_vars, templar, module_compression, async_timeout, become, become_method, become_user, become_password, become_flags, environment, remote_is_local=False):
    """
    Given the source of the module, convert it to a Jinja2 template to insert
    module code and return whether it's a new or old style module.
    """
    module_substyle = module_style = 'old'
    if _is_binary(b_module_data):
        module_substyle = module_style = 'binary'
    elif REPLACER in b_module_data:
        module_style = 'new'
        module_substyle = 'python'
        b_module_data = b_module_data.replace(REPLACER, b'from ansible.module_utils.basic import *')
    elif NEW_STYLE_PYTHON_MODULE_RE.search(b_module_data):
        module_style = 'new'
        module_substyle = 'python'
    elif REPLACER_WINDOWS in b_module_data:
        module_style = 'new'
        module_substyle = 'powershell'
        b_module_data = b_module_data.replace(REPLACER_WINDOWS, b'#Requires -Module Ansible.ModuleUtils.Legacy')
    elif re.search(b'#Requires -Module', b_module_data, re.IGNORECASE) or re.search(b'#Requires -Version', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -OSVersion', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -Powershell', b_module_data, re.IGNORECASE) or re.search(b'#AnsibleRequires -CSharpUtil', b_module_data, re.IGNORECASE):
        module_style = 'new'
        module_substyle = 'powershell'
    elif REPLACER_JSONARGS in b_module_data:
        module_style = 'new'
        module_substyle = 'jsonargs'
    elif b'WANT_JSON' in b_module_data:
        module_substyle = module_style = 'non_native_want_json'
    shebang = None
    if module_style in ('old', 'non_native_want_json', 'binary'):
        return (b_module_data, module_style, shebang)
    output = BytesIO()
    try:
        remote_module_fqn = _get_ansible_module_fqn(module_path)
    except ValueError:
        display.debug('ANSIBALLZ: Could not determine module FQN')
        remote_module_fqn = 'ansible.modules.%s' % module_name
    if module_substyle == 'python':
        date_time = time.gmtime()[:6]
        if date_time[0] < 1980:
            date_string = datetime.datetime(*date_time, tzinfo=datetime.timezone.utc).strftime('%c')
            raise AnsibleError(f'Cannot create zipfile due to pre-1980 configured date: {date_string}')
        params = dict(ANSIBLE_MODULE_ARGS=module_args)
        try:
            python_repred_params = repr(json.dumps(params, cls=AnsibleJSONEncoder, vault_to_text=True))
        except TypeError as e:
            raise AnsibleError('Unable to pass options to module, they must be JSON serializable: %s' % to_native(e))
        try:
            compression_method = getattr(zipfile, module_compression)
        except AttributeError:
            display.warning(u'Bad module compression string specified: %s.  Using ZIP_STORED (no compression)' % module_compression)
            compression_method = zipfile.ZIP_STORED
        lookup_path = os.path.join(C.DEFAULT_LOCAL_TMP, 'ansiballz_cache')
        cached_module_filename = os.path.join(lookup_path, '%s-%s' % (remote_module_fqn, module_compression))
        zipdata = None
        if os.path.exists(cached_module_filename):
            display.debug('ANSIBALLZ: using cached module: %s' % cached_module_filename)
            with open(cached_module_filename, 'rb') as module_data:
                zipdata = module_data.read()
        else:
            if module_name in action_write_locks.action_write_locks:
                display.debug('ANSIBALLZ: Using lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[module_name]
            else:
                display.debug('ANSIBALLZ: Using generic lock for %s' % module_name)
                lock = action_write_locks.action_write_locks[None]
            display.debug('ANSIBALLZ: Acquiring lock')
            with lock:
                display.debug('ANSIBALLZ: Lock acquired: %s' % id(lock))
                if not os.path.exists(cached_module_filename):
                    display.debug('ANSIBALLZ: Creating module')
                    zipoutput = BytesIO()
                    zf = zipfile.ZipFile(zipoutput, mode='w', compression=compression_method)
                    recursive_finder(module_name, remote_module_fqn, b_module_data, zf, date_time)
                    display.debug('ANSIBALLZ: Writing module into payload')
                    _add_module_to_zip(zf, date_time, remote_module_fqn, b_module_data)
                    zf.close()
                    zipdata = base64.b64encode(zipoutput.getvalue())
                    if not os.path.exists(lookup_path):
                        try:
                            os.makedirs(lookup_path)
                        except OSError:
                            if not os.path.exists(lookup_path):
                                raise
                    display.debug('ANSIBALLZ: Writing module')
                    with open(cached_module_filename + '-part', 'wb') as f:
                        f.write(zipdata)
                    display.debug('ANSIBALLZ: Renaming module')
                    os.rename(cached_module_filename + '-part', cached_module_filename)
                    display.debug('ANSIBALLZ: Done creating module')
            if zipdata is None:
                display.debug('ANSIBALLZ: Reading module after lock')
                try:
                    with open(cached_module_filename, 'rb') as f:
                        zipdata = f.read()
                except IOError:
                    raise AnsibleError('A different worker process failed to create module file. Look at traceback for that process for debugging information.')
        zipdata = to_text(zipdata, errors='surrogate_or_strict')
        o_interpreter, o_args = _extract_interpreter(b_module_data)
        if o_interpreter is None:
            o_interpreter = u'/usr/bin/python'
        shebang, interpreter = _get_shebang(o_interpreter, task_vars, templar, o_args, remote_is_local=remote_is_local)
        rlimit_nofile = C.config.get_config_value('PYTHON_MODULE_RLIMIT_NOFILE', variables=task_vars)
        if not isinstance(rlimit_nofile, int):
            rlimit_nofile = int(templar.template(rlimit_nofile))
        if rlimit_nofile:
            rlimit = ANSIBALLZ_RLIMIT_TEMPLATE % dict(rlimit_nofile=rlimit_nofile)
        else:
            rlimit = ''
        coverage_config = os.environ.get('_ANSIBLE_COVERAGE_CONFIG')
        if coverage_config:
            coverage_output = os.environ['_ANSIBLE_COVERAGE_OUTPUT']
            if coverage_output:
                coverage = ANSIBALLZ_COVERAGE_TEMPLATE % dict(coverage_config=coverage_config, coverage_output=coverage_output)
            else:
                coverage = ANSIBALLZ_COVERAGE_CHECK_TEMPLATE
        else:
            coverage = ''
        output.write(to_bytes(ACTIVE_ANSIBALLZ_TEMPLATE % dict(zipdata=zipdata, ansible_module=module_name, module_fqn=remote_module_fqn, params=python_repred_params, shebang=shebang, coding=ENCODING_STRING, date_time=date_time, coverage=coverage, rlimit=rlimit)))
        b_module_data = output.getvalue()
    elif module_substyle == 'powershell':
        shebang = u'#!powershell'
        b_module_data = ps_manifest._create_powershell_wrapper(b_module_data, module_path, module_args, environment, async_timeout, become, become_method, become_user, become_password, become_flags, module_substyle, task_vars, remote_module_fqn)
    elif module_substyle == 'jsonargs':
        module_args_json = to_bytes(json.dumps(module_args, cls=AnsibleJSONEncoder, vault_to_text=True))
        python_repred_args = to_bytes(repr(module_args_json))
        b_module_data = b_module_data.replace(REPLACER_VERSION, to_bytes(repr(__version__)))
        b_module_data = b_module_data.replace(REPLACER_COMPLEX, python_repred_args)
        b_module_data = b_module_data.replace(REPLACER_SELINUX, to_bytes(','.join(C.DEFAULT_SELINUX_SPECIAL_FS)))
        b_module_data = b_module_data.replace(REPLACER_JSONARGS, module_args_json)
        facility = b'syslog.' + to_bytes(task_vars.get('ansible_syslog_facility', C.DEFAULT_SYSLOG_FACILITY), errors='surrogate_or_strict')
        b_module_data = b_module_data.replace(b'syslog.LOG_USER', facility)
    return (b_module_data, module_style, shebang)