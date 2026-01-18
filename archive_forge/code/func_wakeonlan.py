from __future__ import absolute_import, division, print_function
import socket
import struct
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def wakeonlan(module, mac, broadcast, port):
    """ Send a magic Wake-on-LAN packet. """
    mac_orig = mac
    if len(mac) == 12 + 5:
        mac = mac.replace(mac[2], '')
    if len(mac) != 12:
        module.fail_json(msg='Incorrect MAC address length: %s' % mac_orig)
    try:
        int(mac, 16)
    except ValueError:
        module.fail_json(msg='Incorrect MAC address format: %s' % mac_orig)
    data = b''
    padding = ''.join(['FFFFFFFFFFFF', mac * 20])
    for i in range(0, len(padding), 2):
        data = b''.join([data, struct.pack('B', int(padding[i:i + 2], 16))])
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if not module.check_mode:
        try:
            sock.sendto(data, (broadcast, port))
        except socket.error as e:
            sock.close()
            module.fail_json(msg=to_native(e), exception=traceback.format_exc())
    sock.close()