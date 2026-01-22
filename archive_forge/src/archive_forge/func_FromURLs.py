from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import os
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import schemas
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import requests
import six
@staticmethod
def FromURLs(*urls, **kwargs):
    """Loads a snapshot from a series of URLs.

    Args:
      *urls: str, The URLs to the files to load.
      **kwargs: command_path: the command path to include in the User-Agent
        header if the URL is HTTP

    Returns:
      A ComponentSnapshot object.

    Raises:
      URLFetchError: If the URL cannot be fetched.
      TypeError: If an unexpected keyword argument is given.
    """
    current_function_name = ComponentSnapshot.FromURLs.__name__
    unexpected_args = set(kwargs) - set(['command_path'])
    if unexpected_args:
        raise TypeError("{0} got an unexpected keyword argument '{1}'".format(current_function_name, unexpected_args.pop()))
    command_path = kwargs.get('command_path', 'unknown')
    first = urls[0]
    data = [(ComponentSnapshot._DictFromURL(url, command_path, is_extra_repo=url != first), url) for url in urls]
    return ComponentSnapshot._FromDictionary(*data)