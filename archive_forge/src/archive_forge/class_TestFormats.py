import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
class TestFormats(utils.TestCommand):

    def test_json_format(self):
        self.cmd = ShowJson(self.app, None)
        parsed_args = self.check_parser(self.cmd, [], [])
        expected = json.dumps(dict(zip(columns, data)), indent=2)
        self.cmd.run(parsed_args)
        self.assertEqual(expected, self.app.stdout.make_string().rstrip())

    def test_yaml_format(self):
        self.cmd = ShowYaml(self.app, None)
        parsed_args = self.check_parser(self.cmd, [], [])
        expected = yaml.safe_dump(dict(zip(columns, data)), default_flow_style=False)
        self.cmd.run(parsed_args)
        self.assertEqual(expected, self.app.stdout.make_string())

    def test_shell_format(self):
        self.cmd = ShowShell(self.app, None)
        parsed_args = self.check_parser(self.cmd, [], [])
        expected = 'col1="abcde"\ncol2="[\'fg\', \'hi\', \'jk\']"\ncol3="{\'lmnop\': \'qrstu\'}"\n'
        self.cmd.run(parsed_args)
        self.assertEqual(expected, self.app.stdout.make_string())

    def test_value_format(self):
        self.cmd = ShowValue(self.app, None)
        parsed_args = self.check_parser(self.cmd, [], [])
        expected = "abcde\n['fg', 'hi', 'jk']\n{'lmnop': 'qrstu'}\n"
        self.cmd.run(parsed_args)
        self.assertEqual(expected, self.app.stdout.make_string())

    def test_indent_and_truncate(self):
        self.assertIsNone(format_utils.indent_and_truncate(None))
        self.assertIsNone(format_utils.indent_and_truncate(None, truncate=True))
        self.assertEqual('', format_utils.indent_and_truncate(''))
        self.assertEqual('one', format_utils.indent_and_truncate('one'))
        self.assertIsNone(format_utils.indent_and_truncate(None, spaces=2))
        self.assertEqual('', format_utils.indent_and_truncate('', spaces=2))
        self.assertEqual('  one', format_utils.indent_and_truncate('one', spaces=2))
        self.assertEqual('one\ntwo\nthree\nfour\nfive', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive'))
        self.assertEqual('three\nfour\nfive', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive', truncate=True, truncate_limit=3))
        self.assertEqual('  and so on\n  three\n  four\n  five\n  truncated', format_utils.indent_and_truncate('one\ntwo\nthree\nfour\nfive', spaces=2, truncate=True, truncate_limit=3, truncate_prefix='and so on', truncate_postfix='truncated'))

    def test_print_software_deployment_output(self):
        out = io.StringIO()
        format_utils.print_software_deployment_output({'deploy_stdout': ''}, out=out, name='deploy_stdout')
        self.assertEqual('  deploy_stdout: |\n\n', out.getvalue())
        ov = {'deploy_stdout': '', 'deploy_stderr': '1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11', 'deploy_status_code': 0}
        out = io.StringIO()
        format_utils.print_software_deployment_output(ov, out=out, name='deploy_stderr')
        self.assertEqual('  deploy_stderr: |\n    ...\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n    10\n    11\n    (truncated, view all with --long)\n', out.getvalue())
        out = io.StringIO()
        format_utils.print_software_deployment_output(ov, out=out, name='deploy_stderr', long=True)
        self.assertEqual('  deploy_stderr: |\n    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n    10\n    11\n', out.getvalue())