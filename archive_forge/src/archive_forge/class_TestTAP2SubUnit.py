from io import BytesIO, StringIO
from testtools import TestCase
from testtools.compat import _u
from testtools.testresult.doubles import StreamResult
import subunit
class TestTAP2SubUnit(TestCase):
    """Tests for TAP2SubUnit.

    These tests test TAP string data in, and subunit string data out.
    This is ok because the subunit protocol is intended to be stable,
    but it might be easier/pithier to write tests against TAP string in,
    parsed subunit objects out (by hooking the subunit stream to a subunit
    protocol server.
    """

    def setUp(self):
        super().setUp()
        self.tap = StringIO()
        self.subunit = BytesIO()

    def test_skip_entire_file(self):
        self.tap.write(_u('1..0 # Skipped: entire file skipped\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'file skip', 'skip', None, True, 'tap comment', b'Skipped: entire file skipped', True, None, None, None)])

    def test_ok_test_pass(self):
        self.tap.write(_u('ok\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'success', None, False, None, None, True, None, None, None)])

    def test_ok_test_number_pass(self):
        self.tap.write(_u('ok 1\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'success', None, False, None, None, True, None, None, None)])

    def test_ok_test_number_description_pass(self):
        self.tap.write(_u('ok 1 - There is a description\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 - There is a description', 'success', None, False, None, None, True, None, None, None)])

    def test_ok_test_description_pass(self):
        self.tap.write(_u('ok There is a description\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 There is a description', 'success', None, False, None, None, True, None, None, None)])

    def test_ok_SKIP_skip(self):
        self.tap.write(_u('ok # SKIP\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'skip', None, False, None, None, True, None, None, None)])

    def test_ok_skip_number_comment_lowercase(self):
        self.tap.write(_u('ok 1 # skip no samba environment available, skipping compilation\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'skip', None, False, 'tap comment', b'no samba environment available, skipping compilation', True, 'text/plain; charset=UTF8', None, None)])

    def test_ok_number_description_SKIP_skip_comment(self):
        self.tap.write(_u('ok 1 foo  # SKIP Not done yet\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 foo', 'skip', None, False, 'tap comment', b'Not done yet', True, 'text/plain; charset=UTF8', None, None)])

    def test_ok_SKIP_skip_comment(self):
        self.tap.write(_u('ok # SKIP Not done yet\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'skip', None, False, 'tap comment', b'Not done yet', True, 'text/plain; charset=UTF8', None, None)])

    def test_ok_TODO_xfail(self):
        self.tap.write(_u('ok # TODO\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'xfail', None, False, None, None, True, None, None, None)])

    def test_ok_TODO_xfail_comment(self):
        self.tap.write(_u('ok # TODO Not done yet\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'xfail', None, False, 'tap comment', b'Not done yet', True, 'text/plain; charset=UTF8', None, None)])

    def test_bail_out_errors(self):
        self.tap.write(_u('ok 1 foo\n'))
        self.tap.write(_u('Bail out! Lifejacket engaged\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 foo', 'success', None, False, None, None, True, None, None, None), ('status', 'Bail out! Lifejacket engaged', 'fail', None, False, None, None, True, None, None, None)])

    def test_missing_test_at_end_with_plan_adds_error(self):
        self.tap.write(_u('1..3\n'))
        self.tap.write(_u('ok first test\n'))
        self.tap.write(_u('not ok second test\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 first test', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2 second test', 'fail', None, False, None, None, True, None, None, None), ('status', 'test 3', 'fail', None, False, 'tap meta', b'test missing from TAP output', True, 'text/plain; charset=UTF8', None, None)])

    def test_missing_test_with_plan_adds_error(self):
        self.tap.write(_u('1..3\n'))
        self.tap.write(_u('ok first test\n'))
        self.tap.write(_u('not ok 3 third test\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 first test', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2', 'fail', None, False, 'tap meta', b'test missing from TAP output', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 3 third test', 'fail', None, False, None, None, True, None, None, None)])

    def test_missing_test_no_plan_adds_error(self):
        self.tap.write(_u('ok first test\n'))
        self.tap.write(_u('not ok 3 third test\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 first test', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2', 'fail', None, False, 'tap meta', b'test missing from TAP output', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 3 third test', 'fail', None, False, None, None, True, None, None, None)])

    def test_four_tests_in_a_row_trailing_plan(self):
        self.tap.write(_u('ok 1 - first test in a script with trailing plan\n'))
        self.tap.write(_u('not ok 2 - second\n'))
        self.tap.write(_u('ok 3 - third\n'))
        self.tap.write(_u('not ok 4 - fourth\n'))
        self.tap.write(_u('1..4\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 - first test in a script with trailing plan', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2 - second', 'fail', None, False, None, None, True, None, None, None), ('status', 'test 3 - third', 'success', None, False, None, None, True, None, None, None), ('status', 'test 4 - fourth', 'fail', None, False, None, None, True, None, None, None)])

    def test_four_tests_in_a_row_with_plan(self):
        self.tap.write(_u('1..4\n'))
        self.tap.write(_u('ok 1 - first test in a script with a plan\n'))
        self.tap.write(_u('not ok 2 - second\n'))
        self.tap.write(_u('ok 3 - third\n'))
        self.tap.write(_u('not ok 4 - fourth\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 - first test in a script with a plan', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2 - second', 'fail', None, False, None, None, True, None, None, None), ('status', 'test 3 - third', 'success', None, False, None, None, True, None, None, None), ('status', 'test 4 - fourth', 'fail', None, False, None, None, True, None, None, None)])

    def test_four_tests_in_a_row_no_plan(self):
        self.tap.write(_u('ok 1 - first test in a script with no plan at all\n'))
        self.tap.write(_u('not ok 2 - second\n'))
        self.tap.write(_u('ok 3 - third\n'))
        self.tap.write(_u('not ok 4 - fourth\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1 - first test in a script with no plan at all', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2 - second', 'fail', None, False, None, None, True, None, None, None), ('status', 'test 3 - third', 'success', None, False, None, None, True, None, None, None), ('status', 'test 4 - fourth', 'fail', None, False, None, None, True, None, None, None)])

    def test_todo_and_skip(self):
        self.tap.write(_u('not ok 1 - a fail but # TODO but is TODO\n'))
        self.tap.write(_u('not ok 2 - another fail # SKIP instead\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.subunit.seek(0)
        events = StreamResult()
        subunit.ByteStreamToStreamResult(self.subunit).run(events)
        self.check_events([('status', 'test 1 - a fail but', 'xfail', None, False, 'tap comment', b'but is TODO', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 2 - another fail', 'skip', None, False, 'tap comment', b'instead', True, 'text/plain; charset=UTF8', None, None)])

    def test_leading_comments_add_to_next_test_log(self):
        self.tap.write(_u('# comment\n'))
        self.tap.write(_u('ok\n'))
        self.tap.write(_u('ok\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'success', None, False, 'tap comment', b'# comment', True, 'text/plain; charset=UTF8', None, None), ('status', 'test 2', 'success', None, False, None, None, True, None, None, None)])

    def test_trailing_comments_are_included_in_last_test_log(self):
        self.tap.write(_u('ok\n'))
        self.tap.write(_u('ok\n'))
        self.tap.write(_u('# comment\n'))
        self.tap.seek(0)
        result = subunit.TAP2SubUnit(self.tap, self.subunit)
        self.assertEqual(0, result)
        self.check_events([('status', 'test 1', 'success', None, False, None, None, True, None, None, None), ('status', 'test 2', 'success', None, False, 'tap comment', b'# comment', True, 'text/plain; charset=UTF8', None, None)])

    def check_events(self, events):
        self.subunit.seek(0)
        eventstream = StreamResult()
        subunit.ByteStreamToStreamResult(self.subunit).run(eventstream)
        self.assertEqual(events, eventstream._events)