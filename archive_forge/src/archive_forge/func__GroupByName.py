import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def _GroupByName(self, name):
    if 'mainGroup' not in self._properties:
        self.SetProperty('mainGroup', PBXGroup())
    main_group = self._properties['mainGroup']
    group = main_group.GetChildByName(name)
    if group is None:
        group = PBXGroup({'name': name})
        main_group.AppendChild(group)
    return group