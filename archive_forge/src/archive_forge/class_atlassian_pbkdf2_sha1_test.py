import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
class atlassian_pbkdf2_sha1_test(HandlerCase):
    handler = hash.atlassian_pbkdf2_sha1
    known_correct_hashes = [('admin', '{PKCS5S2}c4xaeTQM0lUieMS3V5voiexyX9XhqC2dBd5ecVy60IPksHChwoTAVYFrhsgoq8/p'), (UPASS_WAV, '{PKCS5S2}cE9Yq6Am5tQGdHSHhky2XLeOnURwzaLBG2sur7FHKpvy2u0qDn6GcVGRjlmJoIUy')]
    known_malformed_hashes = ['{PKCS5S2}c4xaeTQM0lUieMS3V5voiexyX9XhqC2dBd5ecVy!0IPksHChwoTAVYFrhsgoq8/p{PKCS5S2}c4xaeTQM0lUieMS3V5voiexyX9XhqC2dBd5ecVy60IPksHChwoTAVYFrhsgoq8/{PKCS5S2}c4xaeTQM0lUieMS3V5voiexyX9XhqC2dBd5ecVy60IPksHChwoTAVYFrhsgoq8/=']