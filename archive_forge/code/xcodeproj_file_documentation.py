import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
Update hash with data's length and contents.

      If the hash were updated only with the value of data, it would be
      possible for clowns to induce collisions by manipulating the names of
      their objects.  By adding the length, it's exceedingly less likely that
      ID collisions will be encountered, intentionally or not.
      