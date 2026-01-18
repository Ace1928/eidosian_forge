from __future__ import with_statement
import hashlib
import warnings
from passlib.utils.compat import u, JYTHON
from passlib.tests.utils import TestCase, hb
test custom prf function