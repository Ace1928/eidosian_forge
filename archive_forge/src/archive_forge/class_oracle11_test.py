from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class oracle11_test(HandlerCase):
    handler = hash.oracle11
    known_correct_hashes = [('abc123', 'S:5FDAB69F543563582BA57894FE1C1361FB8ED57B903603F2C52ED1B4D642'), ('SyStEm123!@#', 'S:450F957ECBE075D2FA009BA822A9E28709FBC3DA82B44D284DDABEC14C42'), ('oracle', 'S:3437FF72BD69E3FB4D10C750B92B8FB90B155E26227B9AB62D94F54E5951'), ('11g', 'S:61CE616647A4F7980AFD7C7245261AF25E0AFE9C9763FCF0D54DA667D4E6'), ('11g', 'S:B9E7556F53500C8C78A58F50F24439D79962DE68117654B6700CE7CC71CF'), ('SHAlala', 'S:2BFCFDF5895014EE9BB2B9BA067B01E0389BB5711B7B5F82B7235E9E182C'), (UPASS_TABLE, 'S:51586343E429A6DF024B8F242F2E9F8507B1096FACD422E29142AA4974B0')]