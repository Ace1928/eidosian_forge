import random
import struct
import zipfile
from hashlib import md5
from twisted.python import filepath, zipstream
from twisted.trial import unittest

        Create a zip file with the given file name and compression scheme.
        