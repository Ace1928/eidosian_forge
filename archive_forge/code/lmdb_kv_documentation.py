from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_native, to_text

        terms contain any number of keys to be retrieved.
        If terms is None, all keys from the database are returned
        with their values, and if term ends in an asterisk, we
        start searching there

        The LMDB database defaults to 'ansible.mdb' if Ansible's
        variable 'lmdb_kv_db' is not set:

              vars:
                - lmdb_kv_db: "jp.mdb"
        