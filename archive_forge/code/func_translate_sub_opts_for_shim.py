from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import re
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def translate_sub_opts_for_shim(sub_opts):
    translated_sub_opts = []
    scope_type_from_flag = {'-g': ChangeType.GROUP, '-p': ChangeType.PROJECT, '-u': ChangeType.USER}
    for flag, value in sub_opts:
        if flag in scope_type_from_flag:
            change = AclChange(value, scope_type=scope_type_from_flag[flag])
            new_value = 'entity={},role={}'.format(change.GetEntity(), change.perm)
            translated_sub_opts.append((flag, new_value))
        elif flag == '-d':
            change = AclDel(value)
            translated_sub_opts.append(('-d', change.identifier))
        else:
            translated_sub_opts.append((flag, value))
    return translated_sub_opts