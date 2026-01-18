from __future__ import absolute_import
import codecs
import serial
        Incremental encode, keep track of digits and emit a byte when a pair
        of hex digits is found. The space is optional unless the error
        handling is defined to be 'strict'.
        