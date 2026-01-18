import unittest as ut
from h5py import h5p, h5f, version
from .common import TestCase
def test_obj_track_times(self):
    """
        tests if the object track times  set/get
        """
    gcid = h5p.create(h5p.GROUP_CREATE)
    gcid.set_obj_track_times(False)
    self.assertEqual(False, gcid.get_obj_track_times())
    gcid.set_obj_track_times(True)
    self.assertEqual(True, gcid.get_obj_track_times())
    dcid = h5p.create(h5p.DATASET_CREATE)
    dcid.set_obj_track_times(False)
    self.assertEqual(False, dcid.get_obj_track_times())
    dcid.set_obj_track_times(True)
    self.assertEqual(True, dcid.get_obj_track_times())
    ocid = h5p.create(h5p.OBJECT_CREATE)
    ocid.set_obj_track_times(False)
    self.assertEqual(False, ocid.get_obj_track_times())
    ocid.set_obj_track_times(True)
    self.assertEqual(True, ocid.get_obj_track_times())