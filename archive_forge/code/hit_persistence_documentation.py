import unittest
import pickle
from .common import MTurkCommon

		It seems the technique used to store and reload the object must
		result in an equivalent object, or subsequent pickles may fail.
		This tests a double-pickle to elicit that error.
		