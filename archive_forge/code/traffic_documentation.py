from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
Update traffic tags.

    Removes and/or clears existing traffic tags as requested. Always adds new
    tags to zero percent targets for the specified revision. Treats a tag
    update as a remove and add.

    Args:
      to_update: A dictionary mapping tag to revision name or 'LATEST' for the
        latest ready revision.
      to_remove: A list of tags to remove.
      clear_others: A boolean indicating whether to clear tags not specified in
        to_update.
    