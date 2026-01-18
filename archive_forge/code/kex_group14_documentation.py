from paramiko.kex_group1 import KexGroup1
from hashlib import sha1, sha256

Standard SSH key exchange ("kex" if you wanna sound cool).  Diffie-Hellman of
2048 bit key halves, using a known "p" prime and "g" generator.
