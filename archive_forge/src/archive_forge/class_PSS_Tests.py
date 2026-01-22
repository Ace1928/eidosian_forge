import unittest
from Cryptodome.Util.py3compat import b, bchr
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.SelfTest.st_common import list_test_cases
from Cryptodome.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Cryptodome.Hash import SHA1, SHA224, SHA256, SHA384, SHA512
from Cryptodome.PublicKey import RSA
from Cryptodome.Signature import pss
from Cryptodome.Signature import PKCS1_PSS
from Cryptodome.Signature.pss import MGF1
class PSS_Tests(unittest.TestCase):
    rsa_key = b'-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAsvI34FgiTK8+txBvmooNGpNwk23YTU51dwNZi5yha3W4lA/Q\nvcZrDalkmD7ekWQwnduxVKa6pRSI13KBgeUOIqJoGXSWhntEtY3FEwvWOHW5AE7Q\njUzTzCiYT6TVaCcpa/7YLai+p6ai2g5f5Zfh4jSawa9uYeuggFygQq4IVW796MgV\nyqxYMM/arEj+/sKz3Viua9Rp9fFosertCYCX4DUTgW0mX9bwEnEOgjSI3pLOPXz1\n8vx+DRZS5wMCmwCUa0sKonLn3cAUPq+sGix7+eo7T0Z12MU8ud7IYVX/75r3cXiF\nPaYE2q8Le0kgOApIXbb+x74x0rNgyIh1yGygkwIDAQABAoIBABz4t1A0pLT6qHI2\nEIOaNz3mwhK0dZEqkz0GB1Dhtoax5ATgvKCFB98J3lYB08IBURe1snOsnMpOVUtg\naBRSM+QqnCUG6bnzKjAkuFP5liDE+oNQv1YpKp9CsUovuzdmI8Au3ewihl+ZTIN2\nUVNYMEOR1b5m+z2SSwWNOYsiJwpBrT7zkpdlDyjat7FiiPhMMIMXjhQFVxURMIcB\njUBtPzGvV/PG90cVDWi1wRGeeP1dDqti/jsnvykQ15KW1MqGrpeNKRmDdTy/Ucl1\nWIoYklKw3U456lgZ/rDTDB818+Tlnk35z4yF7d5ANPM8CKfqOPcnO1BCKVFzf4eq\n54wvUtkCgYEA1Zv2lp06l7rXMsvNtyYQjbFChezRDRnPwZmN4NCdRtTgGG1G0Ryd\nYz6WWoPGqZp0b4LAaaHd3W2GTcpXF8WXMKfMX1W+tMAxMozfsXRKMcHoypwuS5wT\nfJRXJCG4pvd57AB0iVUEJW2we+uGKU5Zxcx//id2nXGCpoRyViIplQsCgYEA1nVC\neHupHChht0Fh4N09cGqZHZzuwXjOUMzR3Vsfz+4WzVS3NvIgN4g5YgmQFOeKwo5y\niRq5yvubcNdFvf85eHWClg0zPAyxJCVUWigCrrOanGEhJo6re4idJvNVzu4Ucg0v\n6B3SJ1HsCda+ZSNz24bSyqRep8A+RoAaoVSFx5kCgYEAn3RvXPs9s+obnqWYiPF3\nRe5etE6Vt2vfNKwFxx6zaR6bsmBQjuUHcABWiHb6I71S0bMPI0tbrWGG8ibrYKl1\nNTLtUvVVCOS3VP7oNTWT9RTFTAnOXU7DFSo+6o/poWn3r36ff6zhDXeWWMr2OXtt\ndEQ1/2lCGEGVv+v61eVmmQUCgYABFHITPTwqwiFL1O5zPWnzyPWgaovhOYSAb6eW\n38CXQXGn8wdBJZL39J2lWrr4//l45VK6UgIhfYbY2JynSkO10ZGow8RARygVMILu\nOUlaK9lZdDvAf/NpGdUAvzTtZ9F+iYZ2OsA2JnlzyzsGM1l//3vMPWukmJk3ral0\nqoJJ8QKBgGRG3eVHnIegBbFVuMDp2NTcfuSuDVUQ1fGAwtPiFa8u81IodJnMk2pq\niXu2+0ytNA/M+SVrAnE2AgIzcaJbtr0p2srkuVM7KMWnG1vWFNjtXN8fAhf/joOv\nD+NmPL/N4uE57e40tbiU/H7KdyZaDt+5QiTmdhuyAe6CBjKsF2jy\n-----END RSA PRIVATE KEY-----'
    msg = b'AAA'
    tag = b'\x00[c5\xd8\xb0\x8b!D\x81\x83\x07\xc0\xdd\xb9\xb4\xb2`\x92\xe7\x02\xf1\xe1P\xea\xc3\xf0\xe3>\xddX5\xdd\x8e\xc5\x89\xef\xf3\xc2\xdc\xfeP\x02\x7f\x12+\xc9\xaf\xbb\xec\xfe\xb0\xa5\xb9\x08\x11P\x8fL\xee5\x9b\xb0k{=_\xd2\x14\xfb\x01R\xb7\xfe\x14}b\x03\x8d5Y\x89~}\xfc\xf2l\xd01-\xbd\xeb\x11\xcdV\x11\xe9l\x19k/o5\xa2\x0f\x15\xe7Q$\t=\xec\x1dAB\x19\xa5P\x9a\xaf\xa3G\x86"\xd6~\xf0<p5\x00\x86\xe0\xf3\x99\xc7+\xcfc,\\\x13)v\xcd\xff\x08o\x90\xc5\xd1\xca\x869\xf45\x1e\xfd\xa2\xf1n\xa3\xa6e\xc5\x11Q\xe4@\xbd\x17\x83x\xc9\x9b\xb5\xc7\xea\x03U\x9b\xa0\xccC\x17\xc9T\x86/\x05\x1c\xc7\x95hC\xf9b1\xbb\x05\xc3\xf0\x9a>j\xfcqkbs\x13\x84b\xe4\xbdm(\xed`\xa4F\xfb\x8f.\xe1\x8c)/_\x9eS\x98\xa4v\xb8\xdc\xfe\xf7/D\x18\x19\xb3T\x97:\xe2\x96s\xe8<\xa2\xb4\xb9\xf8/'

    def test_positive_1(self):
        key = RSA.import_key(self.rsa_key)
        h = SHA256.new(self.msg)
        verifier = pss.new(key)
        verifier.verify(h, self.tag)

    def test_negative_1(self):
        key = RSA.import_key(self.rsa_key)
        h = SHA256.new(self.msg + b'A')
        verifier = pss.new(key)
        tag = bytearray(self.tag)
        self.assertRaises(ValueError, verifier.verify, h, tag)

    def test_negative_2(self):
        key = RSA.import_key(self.rsa_key)
        h = SHA256.new(self.msg)
        verifier = pss.new(key, salt_bytes=1000)
        tag = bytearray(self.tag)
        self.assertRaises(ValueError, verifier.verify, h, tag)