from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class ldap_salted_sha1_test(HandlerCase):
    handler = hash.ldap_salted_sha1
    known_correct_hashes = [('testing123', '{SSHA}0c0blFTXXNuAMHECS4uxrj3ZieMoWImr'), ('secret', '{SSHA}0H+zTv8o4MR4H43n03eCsvw1luG8LdB7'), (UPASS_TABLE, '{SSHA}3yCSD1nLZXznra4N8XzZgAL+s1sQYsx5'), ('test', '{SSHA}P90+qijSp8MJ1tN25j5o1PflUvlqjXHOGeOckw=='), ('test', '{SSHA}/ZMF5KymNM+uEOjW+9STKlfCFj51bg3BmBNCiPHeW2ttbU0='), ('test', '{SSHA}Pfx6Vf48AT9x3FVv8znbo8WQkEVSipHSWovxXmvNWUvp/d/7')]
    known_malformed_hashes = ['{SSHA}ZQK3Yvtvl6wtIRoISgMGPkcWU7Nfq5U=', '{SSHA}P90+qijSp8MJ1tN25j5o1PflUvlqjXHOGeOck', '{SSHA}P90+qijSp8MJ1tN25j5o1PflUvlqjXHOGeOckw=', '{SSHA}P90+qijSp8MJ1tN25j5o1Pf=UvlqjXHOGeOckw==', '{SSHA}P90+qijSp8MJ1tN25j5o1PflUvlqjXHOGeOck===']