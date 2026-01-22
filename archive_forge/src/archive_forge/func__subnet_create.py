import random
import uuid
from openstackclient.tests.functional.network.v2 import common
def _subnet_create(self, cmd, name, is_type_ipv4=True):
    for i in range(4):
        if is_type_ipv4:
            subnet = '.'.join(map(str, (random.randint(0, 223) for _ in range(3)))) + '.0/26'
        else:
            subnet = ':'.join(map(str, (hex(random.randint(0, 65535))[2:] for _ in range(7)))) + ':0/112'
        try:
            cmd_output = self.openstack(cmd + ' ' + subnet + ' ' + name, parse_output=True)
        except Exception:
            if i == 3:
                raise
            pass
        else:
            break
    return cmd_output