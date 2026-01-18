import io
import json
import yaml
from heatclient.common import format_utils
from heatclient.tests.unit.osc import utils
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