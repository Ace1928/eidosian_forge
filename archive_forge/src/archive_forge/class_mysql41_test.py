from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class mysql41_test(HandlerCase):
    handler = hash.mysql41
    known_correct_hashes = [('verysecretpassword', '*2C905879F74F28F8570989947D06A8429FB943E6'), ('12345678123456781234567812345678', '*F9F1470004E888963FB466A5452C9CBD9DF6239C'), ("' OR 1 /*'", '*97CF7A3ACBE0CA58D5391AC8377B5D9AC11D46D9'), ('mypass', '*6C8989366EAF75BB670AD8EA7A7FC1176A95CEF4'), (UPASS_TABLE, '*E7AFE21A9CFA2FC9D15D942AE8FB5C240FE5837B')]
    known_unidentified_hashes = ['*6Z8989366EAF75BB670AD8EA7A7FC1176A95CEF4']