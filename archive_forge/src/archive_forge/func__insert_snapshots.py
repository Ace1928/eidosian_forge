import warnings
from oslotest import base as test_base
import testscenarios
from oslo_utils import imageutils
from unittest import mock
def _insert_snapshots(self, img_info):
    img_info = img_info + ('Snapshot list:',)
    img_info = img_info + ('ID        TAG                 VM SIZE                DATE       VM CLOCK',)
    for i in range(self.snapshot_count):
        img_info = img_info + ('%d        d9a9784a500742a7bb95627bb3aace38    0 2012-08-20 10:52:46 00:00:00.000' % (i + 1),)
    return img_info