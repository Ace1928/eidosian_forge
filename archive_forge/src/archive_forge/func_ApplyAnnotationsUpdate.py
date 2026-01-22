from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ApplyAnnotationsUpdate(args, original_annotations):
    """Applies updates to the list of annotations on a secret.

  Makes no alterations to the original annotations

  Args:
    args (argparse.Namespace): The collection of user-provided arguments.
    original_annotations (list): annotations configured on the secret prior to
      update.

  Returns:
      result (dict): dict of annotations pairs after update.
  """
    if args.IsSpecified('clear_annotations'):
        return {}
    annotations_dict = dict()
    annotations_dict.update({pair.key: pair.value for pair in original_annotations})
    if args.IsSpecified('remove_annotations'):
        for annotation in args.remove_annotations:
            del annotations_dict[annotation]
        new_annotations = dict()
        for annotation_pair in original_annotations:
            if annotation_pair.key in annotations_dict:
                new_annotations[annotation_pair.key] = annotation_pair.value
        return new_annotations
    elif args.IsSpecified('update_annotations'):
        new_annotations = dict()
        annotations_dict.update({pair.key: pair.value for pair in original_annotations})
        new_annotations.update({annotation: metadata for annotation, metadata in args.update_annotations.items()})
        return new_annotations