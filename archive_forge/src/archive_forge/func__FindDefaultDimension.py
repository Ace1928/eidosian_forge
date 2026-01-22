from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
def _FindDefaultDimension(self, dimension_table):
    for dimension in dimension_table:
        if 'default' in dimension.tags:
            return dimension.id
    return None