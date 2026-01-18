from unittest import TestCase
import macaroonbakery.checkers as checkers
def test_serialize(self):
    tests = [('empty namespace', None, b''), ('standard namespace', {'std': ''}, b'std:'), ('several elements', {'std': '', 'http://blah.blah': 'blah', 'one': 'two', 'foo.com/x.v0.1': 'z'}, b'foo.com/x.v0.1:z http://blah.blah:blah one:two std:'), ('sort by URI not by field', {'a': 'one', 'a1': 'two'}, b'a:one a1:two')]
    for test in tests:
        ns = checkers.Namespace(test[1])
        data = ns.serialize_text()
        self.assertEqual(data, test[2])
        self.assertEqual(str(ns), test[2].decode('utf-8'))
    ns1 = checkers.deserialize_namespace(data)
    self.assertEqual(ns1, ns)