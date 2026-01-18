import itertools
import numpy as np
from numba import jit
from numba.core import utils
from numba.tests.support import TestCase, forbid_codegen
from .enum_usecases import *
import unittest

        Test that forbid_codegen() prevents code generation using the @jit
        decorator.
        