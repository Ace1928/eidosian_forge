from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestDefaultTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'cbt'