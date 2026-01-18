import abc
from castellan.common.objects import opaque_data as op_data
from castellan.common.objects import passphrase
from castellan.common.objects import private_key as pri_key
from castellan.common.objects import public_key as pub_key
from castellan.common.objects import symmetric_key as sym_key
from castellan.common.objects import x_509
Lists the KeyManager's configure options.

        KeyManagers should advertise all supported options through this
        method for the purpose of sample generation by oslo-config-generator.
        Each item in the advertised list should be tuple composed by the group
        name and a list of options belonging to that group. None should be used
        as the group name for the DEFAULT group.

        :returns: the list of supported options of a KeyManager.
        