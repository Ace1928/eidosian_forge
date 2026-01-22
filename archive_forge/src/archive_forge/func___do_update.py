from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def __do_update(self, update):
    response = None
    try:
        if self.module.params['protocol'] == 'tcp':
            response = dns.query.tcp(update, self.module.params['server'], timeout=10, port=self.module.params['port'])
        else:
            response = dns.query.udp(update, self.module.params['server'], timeout=10, port=self.module.params['port'])
    except (dns.tsig.PeerBadKey, dns.tsig.PeerBadSignature) as e:
        self.module.fail_json(msg='TSIG update error (%s): %s' % (e.__class__.__name__, to_native(e)))
    except (socket_error, dns.exception.Timeout) as e:
        self.module.fail_json(msg='DNS server error: (%s): %s' % (e.__class__.__name__, to_native(e)))
    return response