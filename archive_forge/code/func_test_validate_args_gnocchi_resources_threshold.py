import argparse
from unittest import mock
import testtools
from aodhclient.v2 import alarm_cli
@mock.patch.object(argparse.ArgumentParser, 'error')
def test_validate_args_gnocchi_resources_threshold(self, mock_arg):
    parser = self.cli_alarm_create.get_parser('aodh alarm create')
    test_parsed_args = parser.parse_args(['--name', 'gnocchi_resources_threshold_test', '--type', 'gnocchi_resources_threshold', '--metric', 'cpu', '--aggregation-method', 'last', '--resource-type', 'generic', '--threshold', '80'])
    self.cli_alarm_create._validate_args(test_parsed_args)
    mock_arg.assert_called_once_with('gnocchi_resources_threshold requires --metric, --threshold, --resource-id, --resource-type and --aggregation-method')