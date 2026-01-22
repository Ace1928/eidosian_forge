import base64
import collections
import datetime
import enum
import struct
import typing
from spnego._asn1 import (
from spnego._text import to_text
class KerberosKDCOptions(enum.IntFlag):
    reserved = 2147483648
    forwardable = 1073741824
    forwarded = 536870912
    proxiable = 268435456
    proxy = 134217728
    allow_postdate = 67108864
    postdated = 33554432
    unused7 = 16777216
    renewable = 8388608
    unused9 = 4194304
    unused10 = 2097152
    opt_hardware_auth = 1048576
    unused12 = 524288
    unused13 = 262144
    constrained_delegation = 131072
    canonicalize = 65536
    request_anonymous = 32768
    unused17 = 16384
    unused18 = 8192
    unused19 = 4096
    unused20 = 2048
    unused21 = 1024
    unused22 = 512
    unused23 = 256
    unused24 = 128
    unused25 = 64
    disable_transited_check = 32
    renewable_ok = 16
    enc_tkt_in_skey = 8
    unused29 = 4
    renew = 2
    validate = 1

    @classmethod
    def native_labels(cls) -> typing.Dict['KerberosKDCOptions', str]:
        return {KerberosKDCOptions.reserved: 'reserved', KerberosKDCOptions.forwardable: 'forwardable', KerberosKDCOptions.forwarded: 'forwarded', KerberosKDCOptions.proxiable: 'proxiable', KerberosKDCOptions.proxy: 'proxy', KerberosKDCOptions.allow_postdate: 'allow-postdate', KerberosKDCOptions.postdated: 'postdated', KerberosKDCOptions.unused7: 'unused7', KerberosKDCOptions.renewable: 'renewable', KerberosKDCOptions.unused9: 'unused9', KerberosKDCOptions.unused10: 'unused10', KerberosKDCOptions.opt_hardware_auth: 'opt-hardware-auth', KerberosKDCOptions.unused12: 'unused12', KerberosKDCOptions.unused13: 'unused13', KerberosKDCOptions.constrained_delegation: 'constrained-delegation', KerberosKDCOptions.canonicalize: 'canonicalize', KerberosKDCOptions.request_anonymous: 'request-anonymous', KerberosKDCOptions.unused17: 'unused17', KerberosKDCOptions.unused18: 'unused18', KerberosKDCOptions.unused19: 'unused19', KerberosKDCOptions.unused20: 'unused20', KerberosKDCOptions.unused21: 'unused21', KerberosKDCOptions.unused22: 'unused22', KerberosKDCOptions.unused23: 'unused23', KerberosKDCOptions.unused24: 'unused24', KerberosKDCOptions.unused25: 'unused25', KerberosKDCOptions.disable_transited_check: 'disable-transited-check', KerberosKDCOptions.renewable_ok: 'renewable-ok', KerberosKDCOptions.enc_tkt_in_skey: 'enc-tkt-in-skey', KerberosKDCOptions.unused29: 'unused29', KerberosKDCOptions.renew: 'renew', KerberosKDCOptions.validate: 'validate'}