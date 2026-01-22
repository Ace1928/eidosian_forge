import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
class dlitz_pbkdf2_sha1_test(HandlerCase):
    handler = hash.dlitz_pbkdf2_sha1
    known_correct_hashes = [('cloadm', '$p5k2$$exec$r1EWMCMk7Rlv3L/RNcFXviDefYa0hlql'), ('gnu', '$p5k2$c$u9HvcT4d$Sd1gwSVCLZYAuqZ25piRnbBEoAesaa/g'), ('dcl', '$p5k2$d$tUsch7fU$nqDkaxMDOFBeJsTSfABsyn.PYUXilHwL'), ('spam', '$p5k2$3e8$H0NX9mT/$wk/sE8vv6OMKuMaqazCJYDSUhWY9YB2J'), (UPASS_WAV, '$p5k2$$KosHgqNo$9mjN8gqjt02hDoP0c2J0ABtLIwtot8cQ')]