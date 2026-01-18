from __future__ import (absolute_import, division, print_function)
import re
def uldap():
    """Return a configured univention uldap object"""

    def construct():
        try:
            secret_file = open('/etc/ldap.secret', 'r')
            bind_dn = 'cn=admin,{0}'.format(base_dn())
        except IOError:
            secret_file = open('/etc/machine.secret', 'r')
            bind_dn = config_registry()['ldap/hostdn']
        pwd_line = secret_file.readline()
        pwd = re.sub('\n', '', pwd_line)
        import univention.admin.uldap
        return univention.admin.uldap.access(host=config_registry()['ldap/master'], base=base_dn(), binddn=bind_dn, bindpw=pwd, start_tls=1)
    return _singleton('uldap', construct)