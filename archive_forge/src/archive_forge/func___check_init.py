from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_bytes
import re
import os
def __check_init(self):
    if self.__filesystems is None:
        self.__filesystems = dict()
        for f in self.__provider.get_filesystems():
            uuid = f['uuid']
            self.__filesystems[uuid] = BtrfsFilesystem(f, self.__provider, self.__module)