from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
class ListItemsView(collections_abc.ItemsView):

    def __iter__(self):
        for item in self._mapping._m:
            if self._mapping._filter(item):
                yield (getattr(item, self._mapping._key_field), item)