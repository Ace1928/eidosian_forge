import unittest
from Cryptodome.Cipher import DES
 Ronald L. Rivest's DES test, see
        http://people.csail.mit.edu/rivest/Destest.txt
    ABSTRACT
    --------

    We present a simple way to test the correctness of a DES implementation:
    Use the recurrence relation:

        X0      =       9474B8E8C73BCA7D (hexadecimal)

        X(i+1)  =       IF  (i is even)  THEN  E(Xi,Xi)  ELSE  D(Xi,Xi)

    to compute a sequence of 64-bit values:  X0, X1, X2, ..., X16.  Here
    E(X,K)  denotes the DES encryption of  X  using key  K, and  D(X,K)  denotes
    the DES decryption of  X  using key  K.  If you obtain

        X16     =       1B1A2DDB4C642438

    your implementation does not have any of the 36,568 possible single-fault
    errors described herein.
    