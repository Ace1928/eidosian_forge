import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
Extension functions should propagate exceptions.

        Either they should return an object, have an 'except' clause, or
        have a "# cannot_raise" to indicate that we've audited them and
        defined them as not raising exceptions.
        