from __future__ import (absolute_import, division, print_function)
import re
def umc_module_for_add(module, container_dn, superordinate=None):
    """Returns an UMC module object prepared for creating a new entry.

    The module is a module specification according to the udm commandline.
    Example values are:
        * users/user
        * shares/share
        * groups/group

    The container_dn MUST be the dn of the container (not of the object to
    be created itself!).
    """
    mod = module_by_name(module)
    position = position_base_dn()
    position.setDn(container_dn)
    obj = mod.object(config(), uldap(), position, superordinate=superordinate)
    obj.open()
    return obj