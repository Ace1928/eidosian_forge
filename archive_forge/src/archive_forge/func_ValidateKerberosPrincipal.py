from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateKerberosPrincipal(kerberos_principal):
    pattern = re.compile('^(.+)/(.+)@(.+)$')
    if not pattern.match(kerberos_principal):
        raise exceptions.BadArgumentException('--kerberos-principal', 'Kerberos Principal {0} does not match ReGeX {1}.'.format(kerberos_principal, pattern))
    return kerberos_principal