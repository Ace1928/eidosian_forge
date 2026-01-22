from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from six.moves import map  # pylint: disable=redefined-builtin
def ParseGroups(resource_class, groups):
    return list(map(resource_class.AdvertisedGroupsValueListEntryValuesEnum, groups))