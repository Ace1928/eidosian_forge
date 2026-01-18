from unittest.mock import patch
from pyct.report import report
@patch('builtins.print')
@patch('platform.platform')
def test_report_gives_system_version(mock_platform, mock_print):
    mock_platform.side_effect = ['Darwin-19.2.0', 'Darwin-19.2.0-x86_64-i386-64bit']
    report('system')
    mock_print.assert_called_with('system=Darwin-19.2.0           # OS: Darwin-19.2.0-x86_64-i386-64bit')