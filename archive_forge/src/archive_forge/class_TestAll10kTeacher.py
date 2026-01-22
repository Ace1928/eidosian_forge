from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestAll10kTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'babi:all10k'