import re
import unittest
from wsme import exc
from wsme import types
class B1(types.Base):
    b2 = types.wsattr('B2')