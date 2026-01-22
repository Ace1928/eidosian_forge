from __future__ import (absolute_import, division, print_function)
import os
import subprocess
from subprocess import PIPE
from subprocess import Popen
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.utils.display import Display
class CyberarkPassword:

    def __init__(self, appid=None, query=None, output=None, **kwargs):
        self.appid = appid
        self.query = query
        self.output = output
        self.extra_parms = []
        for key, value in kwargs.items():
            self.extra_parms.append('-p')
            self.extra_parms.append('%s=%s' % (key, value))
        if self.appid is None:
            raise AnsibleError('CyberArk Error: No Application ID specified')
        if self.query is None:
            raise AnsibleError('CyberArk Error: No Vault query specified')
        if self.output is None:
            self.output = 'password'
        else:
            self.output = self.output.lower()
        self.b_delimiter = b'@#@'

    def get(self):
        result_dict = {}
        try:
            all_parms = [CLIPASSWORDSDK_CMD, 'GetPassword', '-p', 'AppDescs.AppID=%s' % self.appid, '-p', 'Query=%s' % self.query, '-o', self.output, '-d', self.b_delimiter]
            all_parms.extend(self.extra_parms)
            b_credential = b''
            b_all_params = [to_bytes(v) for v in all_parms]
            tmp_output, tmp_error = Popen(b_all_params, stdout=PIPE, stderr=PIPE, stdin=PIPE).communicate()
            if tmp_output:
                b_credential = to_bytes(tmp_output)
            if tmp_error:
                raise AnsibleError('ERROR => %s ' % tmp_error)
            if b_credential and b_credential.endswith(b'\n'):
                b_credential = b_credential[:-1]
            output_names = self.output.split(',')
            output_values = b_credential.split(self.b_delimiter)
            for i in range(len(output_names)):
                if output_names[i].startswith('passprops.'):
                    if 'passprops' not in result_dict:
                        result_dict['passprops'] = {}
                    output_prop_name = output_names[i][10:]
                    result_dict['passprops'][output_prop_name] = to_native(output_values[i])
                else:
                    result_dict[output_names[i]] = to_native(output_values[i])
        except subprocess.CalledProcessError as e:
            raise AnsibleError(e.output)
        except OSError as e:
            raise AnsibleError('ERROR - AIM not installed or clipasswordsdk not in standard location. ERROR=(%s) => %s ' % (to_text(e.errno), e.strerror))
        return [result_dict]