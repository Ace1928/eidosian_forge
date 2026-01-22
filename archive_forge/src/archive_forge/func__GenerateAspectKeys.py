from __future__ import annotations
from typing import Any, Dict, List
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.dataplex import parsers as dataplex_parsers
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _GenerateAspectKeys(args: parser_extensions.Namespace, *, remove_aspects_arg_name: str, update_aspects_arg_name: str) -> List[str]:
    """Generate a list of unique aspect keys to be updated or removed.

  This will be used along with the update_mask for updating an Entry. This list
  is populated based on `--update-aspects` and `--remove-aspects` arguments
  (or `--aspects` in case of specialized command like `update-aspects`).

  Args:
    args: The arguments provided to the command.
    remove_aspects_arg_name: The name of the argument that contains the aspect
      keys to be removed.
    update_aspects_arg_name: The name of the argument that contains aspect
      contents to be added or updated.

  Returns:
    A sorted list of unique aspect keys to be updated or removed. Or empty list
    if neither `--update-aspects`, `--remove-aspects` or `--aspects` are
    provided to the command.
  """
    keys = set()
    if args.IsKnownAndSpecified(update_aspects_arg_name):
        keys.update(map(lambda aspect: aspect.key, args.GetValue(update_aspects_arg_name).additionalProperties))
    if args.IsKnownAndSpecified(remove_aspects_arg_name):
        keys.update(args.GetValue(remove_aspects_arg_name))
    return sorted(keys)