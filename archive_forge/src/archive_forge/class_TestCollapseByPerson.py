from ...revision import Revision
from ...tests import TestCase, TestCaseWithTransport
from .cmds import collapse_by_person, get_revisions_and_committers
class TestCollapseByPerson(TestCase):

    def test_no_conflicts(self):
        revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Bar <bar@example.com>'), Revision('3', {}, committer='Bar <bar@example.com>')]
        foo = ('Foo', 'foo@example.com')
        bar = ('Bar', 'bar@example.com')
        committers = {foo: foo, bar: bar}
        info = collapse_by_person(revisions, committers)
        self.assertEqual(2, info[0][0])
        self.assertEqual({'bar@example.com': 2}, info[0][2])
        self.assertEqual({'Bar': 2}, info[0][3])

    def test_different_email(self):
        revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Foo <bar@example.com>'), Revision('3', {}, committer='Foo <bar@example.com>')]
        foo = ('Foo', 'foo@example.com')
        bar = ('Foo', 'bar@example.com')
        committers = {foo: foo, bar: foo}
        info = collapse_by_person(revisions, committers)
        self.assertEqual(3, info[0][0])
        self.assertEqual({'foo@example.com': 1, 'bar@example.com': 2}, info[0][2])
        self.assertEqual({'Foo': 3}, info[0][3])

    def test_different_name(self):
        revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Bar <foo@example.com>'), Revision('3', {}, committer='Bar <foo@example.com>')]
        foo = ('Foo', 'foo@example.com')
        bar = ('Bar', 'foo@example.com')
        committers = {foo: foo, bar: foo}
        info = collapse_by_person(revisions, committers)
        self.assertEqual(3, info[0][0])
        self.assertEqual({'foo@example.com': 3}, info[0][2])
        self.assertEqual({'Foo': 1, 'Bar': 2}, info[0][3])

    def test_different_name_case(self):
        revisions = [Revision('1', {}, committer='Foo <foo@example.com>'), Revision('2', {}, committer='Foo <foo@example.com>'), Revision('3', {}, committer='FOO <bar@example.com>')]
        foo = ('Foo', 'foo@example.com')
        FOO = ('FOO', 'bar@example.com')
        committers = {foo: foo, FOO: foo}
        info = collapse_by_person(revisions, committers)
        self.assertEqual(3, info[0][0])
        self.assertEqual({'foo@example.com': 2, 'bar@example.com': 1}, info[0][2])
        self.assertEqual({'Foo': 2, 'FOO': 1}, info[0][3])