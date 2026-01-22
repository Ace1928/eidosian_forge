from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _get_digest_from_x509_file(module, pem_certificate_file, openssl_bin):
    """ Read a X509 certificate file and output sha256 digest using openssl """
    dummy, tmp_certificate = tempfile.mkstemp()
    module.add_cleanup_file(tmp_certificate)
    _get_first_certificate_from_x509_file(module, pem_certificate_file, tmp_certificate, openssl_bin)
    dgst_cmd = [openssl_bin, 'dgst', '-r', '-sha256', tmp_certificate]
    dgst_rc, dgst_stdout, dgst_stderr = module.run_command(dgst_cmd, check_rc=False)
    if dgst_rc != 0:
        module.fail_json(msg='Internal module failure, cannot compute digest for certificate, error: %s' % dgst_stderr, rc=dgst_rc, cmd=dgst_cmd)
    return dgst_stdout.split(' ')[0]