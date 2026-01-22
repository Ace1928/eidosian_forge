from __future__ import absolute_import, division, print_function
import os
import tempfile
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.six.moves.urllib.request import getproxies
def _get_first_certificate_from_x509_file(module, pem_certificate_file, pem_certificate_output, openssl_bin):
    """ Read a X509 certificate chain file and output the first certificate in the list """
    extract_cmd = [openssl_bin, 'x509', '-in', pem_certificate_file, '-out', pem_certificate_output]
    extract_rc, dummy, extract_stderr = module.run_command(extract_cmd, check_rc=False)
    if extract_rc != 0:
        extract_cmd += ['-inform', 'der']
        extract_rc, dummy, extract_stderr = module.run_command(extract_cmd, check_rc=False)
        if extract_rc != 0:
            module.fail_json(msg='Internal module failure, cannot extract certificate, error: %s' % extract_stderr, rc=extract_rc, cmd=extract_cmd)
    return extract_rc