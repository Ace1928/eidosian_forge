from __future__ import (absolute_import, division, print_function)
from ansible_collections.netapp.ontap.plugins.module_utils.rest_generic import get_one_record
 returns a tuple (uuid, error)
        when module is set and an error is found, fails the module and exit
        when error_on_none IS SET, force an error if vserver is not found
    