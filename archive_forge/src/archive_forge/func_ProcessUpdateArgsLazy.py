from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
import six
def ProcessUpdateArgsLazy(args, labels_cls, orig_labels_thunk):
    """Returns the result of applying the diff constructed from args.

  Lazily fetches the original labels value if needed.

  Args:
    args: argparse.Namespace, the parsed arguments with update_labels,
      remove_labels, and clear_labels
    labels_cls: type, the LabelsValue class for the new labels.
    orig_labels_thunk: callable, a thunk which will return the original labels
      object (of type LabelsValue) when evaluated.

  Returns:
    UpdateResult: the result of applying the diff.

  """
    diff = Diff.FromUpdateArgs(args)
    orig_labels = orig_labels_thunk() if diff.MayHaveUpdates() else None
    return diff.Apply(labels_cls, orig_labels)