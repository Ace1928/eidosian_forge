import unittest
class TestHashing(object):
    """Used as a mixin for TestCase"""

    def test_hash(self):
        for obj_1, obj_2 in self.eq_pairs:
            try:
                if not hash(obj_1) == hash(obj_2):
                    self.fail('%r and %r do not hash equal' % (obj_1, obj_2))
            except Exception as e:
                self.fail('Problem hashing %r and %r: %s' % (obj_1, obj_2, e))
        for obj_1, obj_2 in self.ne_pairs:
            try:
                if hash(obj_1) == hash(obj_2):
                    self.fail("%s and %s hash equal, but shouldn't" % (obj_1, obj_2))
            except Exception as e:
                self.fail('Problem hashing %s and %s: %s' % (obj_1, obj_2, e))