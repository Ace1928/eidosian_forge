from unittest.mock import patch
from pyct.report import report
@patch('builtins.print')
@patch('subprocess.check_output')
def test_report_gives_python_version(mock_check_output, mock_print):
    mock_check_output.side_effect = [b'/mock/opt/anaconda3/envs/pyct/bin/python\n', b'Python 3.7.7\n']
    report('python')
    mock_print.assert_called_with('python=3.7.7                   # /mock/opt/anaconda3/envs/pyct/bin/python')