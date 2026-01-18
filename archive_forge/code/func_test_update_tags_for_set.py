import argparse
import sys
from unittest import mock
from osc_lib.tests import utils as test_utils
from osc_lib.utils import tags
def test_update_tags_for_set(self):
    mock_client = mock.MagicMock()
    mock_obj = mock.MagicMock()
    mock_parsed_args = mock.MagicMock()
    mock_parsed_args.no_tag = True
    mock_parsed_args.tags = ['tag1']
    mock_obj.tags = None
    tags.update_tags_for_set(mock_client, mock_obj, mock_parsed_args)
    mock_client.set_tags.assert_called_once_with(mock_obj, list(mock_parsed_args.tags))
    mock_client.set_tags.reset_mock()
    mock_parsed_args.no_tag = False
    mock_parsed_args.tags = ['tag1']
    mock_obj.tags = ['tag2']
    expected_list = ['tag1', 'tag2']
    tags.update_tags_for_set(mock_client, mock_obj, mock_parsed_args)
    mock_client.set_tags.assert_called_once_with(mock_obj, expected_list)
    mock_client.set_tags.reset_mock()
    mock_parsed_args.no_tag = False
    mock_parsed_args.tags = None
    mock_obj.tags = ['tag2']
    tags.update_tags_for_set(mock_client, mock_obj, mock_parsed_args)
    mock_client.set_tags.assert_not_called()