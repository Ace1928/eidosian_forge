import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
def test_value_format(self):
    self.cmd = ShowValue(self.app, None)
    parsed_args = self.check_parser(self.cmd, [], [])
    expected = "abcde\n['fg', 'hi', 'jk']\n{'lmnop': 'qrstu'}\n"
    self.cmd.run(parsed_args)
    self.assertEqual(expected, self.app.stdout.make_string())