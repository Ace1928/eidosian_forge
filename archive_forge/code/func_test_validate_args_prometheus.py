import argparse
from unittest import mock
import testtools
from aodhclient.v2 import alarm_cli
@mock.patch.object(argparse.ArgumentParser, 'error')
def test_validate_args_prometheus(self, mock_arg):
    parser = self.cli_alarm_create.get_parser('aodh alarm create')
    test_parsed_args = parser.parse_args(['--name', 'prom_test', '--type', 'prometheus', '--comparison-operator', 'gt', '--threshold', '666'])
    self.cli_alarm_create._validate_args(test_parsed_args)
    mock_arg.assert_called_once_with('Prometheus alarm requires --query and --threshold parameters.')