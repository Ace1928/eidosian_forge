import unittest
from numba.tests.support import TestCase, skip_unless_typeguard

Tests to ensure that typeguard is working as expected.
This mostly contains negative tests as proof that typeguard can catch errors.
