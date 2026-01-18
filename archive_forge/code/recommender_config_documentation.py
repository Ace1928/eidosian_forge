from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.recommender import base
from googlecloudsdk.api_lib.recommender import flag_utils
Updates a RecommenderConfig.

    Args:
      config_name: str, the name of the config being retrieved.
      args: argparse.Namespace, The arguments that the command was invoked with.

    Returns:
      The updated RecommenderConfig message.
    Raises:
      Exception: If nothing is updated.
    