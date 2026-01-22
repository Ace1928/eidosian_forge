from parlai.utils.testing import AutoTeacherTest  # noqa: F401
import unittest
class TestBlendedSkillTalkTeacher(unittest.TestCase, AutoTeacherTest):
    task = 'style_gen:labeled_blended_skill_talk'