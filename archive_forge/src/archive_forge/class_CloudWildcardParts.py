from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import fnmatch
import heapq
import os
import pathlib
import re
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors as command_errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.core import log
from googlecloudsdk.core.util import debug_output
import six
class CloudWildcardParts:
    """Different parts of the wildcard string used for querying and filtering."""

    def __init__(self, prefix, filter_pattern, delimiter, suffix):
        """Initialize the CloudWildcardParts object.

    Args:
      prefix (str): The prefix string to be passed to the API request.
        This is the substring before the first occurrance of the wildcard.
      filter_pattern (str): The pattern to be used to filter out the results
        returned by the list_objects call. This is a substring starting from
        the first occurance of the wildcard upto the first delimiter.
      delimiter (str): The delimiter to be passed to the api request.
      suffix (str): The substirng after the first delimiter in the wildcard.
    """
        self.prefix = prefix
        self.filter_pattern = filter_pattern
        self.delimiter = delimiter
        self.suffix = suffix

    @classmethod
    def from_string(cls, string, delimiter=storage_url.CloudUrl.CLOUD_URL_DELIM):
        """Create a CloudWildcardParts instance from a string.

    Args:
      string (str): String that needs to be splitted into different parts.
      delimiter (str): The delimiter to be used for splitting the string.

    Returns:
      WildcardParts object.
    """
        prefix, wildcard_string = _split_on_wildcard(string)
        filter_pattern, _, suffix = wildcard_string.partition(delimiter)
        if '**' in filter_pattern:
            delimiter = None
            filter_pattern = wildcard_string
            suffix = None
        return cls(prefix, filter_pattern, delimiter, suffix)

    def __repr__(self):
        return debug_output.generic_repr(self)