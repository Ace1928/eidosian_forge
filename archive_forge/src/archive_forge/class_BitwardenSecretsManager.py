from __future__ import (absolute_import, division, print_function)
from subprocess import Popen, PIPE
from ansible.errors import AnsibleLookupError
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.ajson import AnsibleJSONDecoder
from ansible.plugins.lookup import LookupBase
class BitwardenSecretsManager(object):

    def __init__(self, path='bws'):
        self._cli_path = path

    @property
    def cli_path(self):
        return self._cli_path

    def _run(self, args, stdin=None):
        p = Popen([self.cli_path] + args, stdout=PIPE, stderr=PIPE, stdin=PIPE)
        out, err = p.communicate(stdin)
        rc = p.wait()
        return (to_text(out, errors='surrogate_or_strict'), to_text(err, errors='surrogate_or_strict'), rc)

    def get_secret(self, secret_id, bws_access_token):
        """Get and return the secret with the given secret_id.
        """
        params = ['--color', 'no', '--access-token', bws_access_token, 'get', 'secret', secret_id]
        out, err, rc = self._run(params)
        if rc != 0:
            raise BitwardenSecretsManagerException(to_text(err))
        return AnsibleJSONDecoder().raw_decode(out)[0]