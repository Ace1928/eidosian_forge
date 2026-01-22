from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import operator
from six.moves import map  # pylint: disable=redefined-builtin
def ParseMode(resource_class, mode):
    return resource_class.AdvertiseModeValueValuesEnum(mode)