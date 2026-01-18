import base64
import os
from textwrap import dedent
from twisted.conch.test import keydata
from twisted.python import randbytes
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_reprPublicRSA(self):
    """
        The repr of a L{keys.Key} contains all of the RSA components for an RSA
        public key.
        """
    self.assertEqual(repr(keys.Key(self.rsaObj).public()), '<RSA Public Key (2048 bits)\nattr e:\n\t01:00:01\nattr n:\n\t00:d5:6a:ac:78:23:d6:d6:1b:ec:25:a1:50:c4:77:\n\t63:50:84:45:01:55:42:14:2a:2a:e0:d0:60:ee:d4:\n\te9:a3:ad:4a:fa:39:06:5e:84:55:75:5f:00:36:bf:\n\t6f:aa:2a:3f:83:26:37:c1:69:2e:5b:fd:f0:f3:d2:\n\t7d:d6:98:cd:3a:40:78:d5:ca:a8:18:c0:11:93:24:\n\t09:0c:81:4c:8f:f7:9c:ed:13:16:6a:a4:04:e9:49:\n\t77:c3:e4:55:64:b3:79:68:9e:2c:08:eb:ac:e8:04:\n\t2d:21:77:05:a7:8e:ef:53:30:0d:a5:e5:bb:3d:6a:\n\te2:09:36:6f:fd:34:d3:7d:6f:46:ff:87:da:a9:29:\n\t27:aa:ff:ad:f5:85:e6:3e:1a:b8:7a:1d:4a:b1:ea:\n\tc0:5a:f7:30:df:1f:c2:a4:e4:ef:3f:91:49:96:40:\n\td5:19:77:2d:37:c3:5e:ec:9d:a6:3a:44:a5:c2:a4:\n\t29:dd:d5:ba:9c:3d:45:b3:c6:2c:18:64:d5:ba:3d:\n\tdf:ab:7f:cd:42:ac:a7:f1:18:0b:a0:58:15:62:0b:\n\ta4:2a:6e:43:c3:e4:04:9f:35:a3:47:8e:46:ed:33:\n\ta5:65:bd:bc:3b:29:6e:02:0b:57:df:74:e8:13:b4:\n\t37:35:7e:83:5f:20:26:60:a6:dc:ad:8b:c6:6c:79:\n\t98:f7>')