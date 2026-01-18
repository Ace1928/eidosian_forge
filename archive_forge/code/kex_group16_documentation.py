from paramiko.kex_group1 import KexGroup1
from hashlib import sha512

Standard SSH key exchange ("kex" if you wanna sound cool).  Diffie-Hellman of
4096 bit key halves, using a known "p" prime and "g" generator.
