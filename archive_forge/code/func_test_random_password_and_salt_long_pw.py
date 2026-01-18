import re
import unittest
from binascii import unhexlify
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Cryptodome.Cipher import AES, DES3
from Cryptodome.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
from Cryptodome.Protocol.KDF import _bcrypt_decode
def test_random_password_and_salt_long_pw(self):
    tvs = [(b'^Q&"]A`%/A(BVGt>QaX0M-#<Q148&f', 4, b'vrRP5vQxyD4LrqiLd/oWRO', b'$2a$04$vrRP5vQxyD4LrqiLd/oWROgrrGINsw3gb4Ga5x2sn01jNmiLVECl6'), (b'nZa!rRf\\U;OL;R?>1ghq_+":Y0CRmY', 5, b'YuQvhokOGVnevctykUYpKu', b'$2a$05$YuQvhokOGVnevctykUYpKutZD2pWeGGYn3auyLOasguMY3/0BbIyq'), (b"F%uN/j>[GuB7-jB'_Yj!Tnb7Y!u^6)", 6, b'5L3vpQ0tG9O7k5gQ8nAHAe', b'$2a$06$5L3vpQ0tG9O7k5gQ8nAHAe9xxQiOcOLh8LGcI0PLWhIznsDt.S.C6'), (b'Z>BobP32ub"Cfe*Q<<WUq3rc=[GJr-', 7, b'hp8IdLueqE6qFh1zYycUZ.', b'$2a$07$hp8IdLueqE6qFh1zYycUZ.twmUH8eSTPQAEpdNXKMlwms9XfKqfea'), (b"Ik&8N['7*[1aCc1lOm8\\jWeD*H$eZM", 8, b'2ANDTYCB9m7vf0Prh7rSru', b'$2a$08$2ANDTYCB9m7vf0Prh7rSrupqpO3jJOkIz2oW/QHB4lCmK7qMytGV6'), (b'O)=%3[E$*q+>-q-=tRSjOBh8\\mLNW.', 9, b'nArqOfdCsD9kIbVnAixnwe', b'$2a$09$nArqOfdCsD9kIbVnAixnwe6s8QvyPYWtQBpEXKir2OJF9/oNBsEFe'), (b'/MH51`!BP&0tj3%YCA;Xk%e3S`o\\EI', 10, b'ePiAc.s.yoBi3B6p1iQUCe', b'$2a$10$ePiAc.s.yoBi3B6p1iQUCezn3mraLwpVJ5XGelVyYFKyp5FZn/y.u'), (b'ptAP"mcg6oH.";c0U2_oll.OKi<!ku', 12, b'aroG/pwwPj1tU5fl9a9pkO', b'$2a$12$aroG/pwwPj1tU5fl9a9pkO4rydAmkXRj/LqfHZOSnR6LGAZ.z.jwa')]
    for idx, (password, cost, salt64, result) in enumerate(tvs):
        x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
        self.assertEqual(x, result)
        bcrypt_check(password, result)