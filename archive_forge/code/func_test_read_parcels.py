import io
from os.path import dirname
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version
import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory
@needs_nibabel_data('nitest-cifti2')
def test_read_parcels():
    img = ci.Cifti2Image.from_filename(DATA_FILE4)
    parcel_mapping = img.header.matrix.get_index_map(1)
    expected_parcels = [('MEDIAL.WALL', ((719, 20, 28550), (810, 21, 28631))), ('BA2_FRB08', ((516, 6757, 17888), (461, 6757, 17887))), ('BA1_FRB08', ((211, 5029, 17974), (214, 3433, 17934))), ('BA3b_FRB08', ((444, 3436, 18065), (397, 3436, 18065))), ('BA4p_FRB08', ((344, 3445, 18164), (371, 3443, 18175))), ('BA3a_FRB08', ((290, 3441, 18140), (289, 3440, 18140))), ('BA4a_FRB08', ((471, 3446, 18181), (455, 3446, 19759))), ('BA6_FRB08', ((1457, 2, 30951), (1400, 2, 30951))), ('BA17_V1_FRB08', ((629, 23155, 25785), (635, 23155, 25759))), ('BA45_FRB08', ((245, 10100, 18774), (214, 10103, 18907))), ('BA44_FRB08', ((226, 10118, 19240), (273, 10119, 19270))), ('hOc5_MT_FRB08', ((104, 15019, 23329), (80, 15023, 23376))), ('BA18_V2_FRB08', ((702, 95, 25902), (651, 98, 25903))), ('V3A_SHM07', ((82, 4, 25050), (82, 4, 25050))), ('V3B_SHM07', ((121, 13398, 23303), (121, 13398, 23303))), ('LO1_KPO10', ((54, 15007, 23543), (54, 15007, 23543))), ('LO2_KPO10', ((79, 15013, 23636), (79, 15013, 23636))), ('PITd_KPO10', ((53, 15018, 23769), (65, 15018, 23769))), ('PITv_KPO10', ((72, 23480, 23974), (72, 23480, 23974))), ('OP1_BSW08', ((470, 8421, 18790), (470, 8421, 18790))), ('OP2_BSW08', ((67, 10, 31060), (67, 10, 31060))), ('OP3_BSW08', ((119, 10137, 18652), (119, 10137, 18652))), ('OP4_BSW08', ((191, 16613, 19429), (192, 16613, 19429))), ('IPS1_SHM07', ((54, 11775, 14496), (54, 11775, 14496))), ('IPS2_SHM07', ((71, 11771, 14587), (71, 11771, 14587))), ('IPS3_SHM07', ((114, 11764, 14783), (114, 11764, 14783))), ('IPS4_SHM07', ((101, 11891, 12653), (101, 11891, 12653))), ('V7_SHM07', ((140, 11779, 14002), (140, 11779, 14002))), ('V4v_SHM07', ((81, 23815, 24557), (90, 23815, 24557))), ('V3d_KPO10', ((90, 23143, 25192), (115, 23143, 25192))), ('14c_OFP03', ((22, 19851, 21311), (22, 19851, 21311))), ('13a_OFP03', ((20, 20963, 21154), (20, 20963, 21154))), ('47s_OFP03', ((211, 10182, 20343), (211, 10182, 20343))), ('14r_OFP03', ((54, 21187, 21324), (54, 21187, 21324))), ('13m_OFP03', ((103, 20721, 21075), (103, 20721, 21075))), ('13l_OFP03', ((101, 20466, 20789), (101, 20466, 20789))), ('32pl_OFP03', ((14, 19847, 21409), (14, 19847, 21409))), ('25_OFP03', ((8, 19844, 27750), (8, 19844, 27750))), ('47m_OFP03', ((200, 10174, 20522), (200, 10174, 20522))), ('47l_OFP03', ((142, 10164, 19969), (160, 10164, 19969))), ('Iai_OFP03', ((153, 10188, 20199), (153, 10188, 20199))), ('10r_OFP03', ((138, 19811, 28267), (138, 19811, 28267))), ('11m_OFP03', ((92, 20850, 21165), (92, 20850, 21165))), ('11l_OFP03', ((200, 20275, 21029), (200, 20275, 21029))), ('47r_OFP03', ((259, 10094, 20535), (259, 10094, 20535))), ('10m_OFP03', ((102, 19825, 21411), (102, 19825, 21411))), ('Iam_OFP03', ((15, 20346, 20608), (15, 20346, 20608))), ('Ial_OFP03', ((89, 10194, 11128), (89, 10194, 11128))), ('24_OFP03', ((39, 19830, 28279), (36, 19830, 28279))), ('Iapm_OFP03', ((7, 20200, 20299), (7, 20200, 20299))), ('10p_OFP03', ((480, 19780, 28640), (480, 19780, 28640))), ('V6_PHG06', ((72, 12233, 12869), (72, 12233, 12869))), ('ER_FRB08', ((103, 21514, 26470), (103, 21514, 26470))), ('13b_OFP03', ((60, 21042, 21194), (71, 21040, 21216)))]
    assert img.shape[1] == len(expected_parcels)
    assert len(list(parcel_mapping.parcels)) == len(expected_parcels)
    for (name, expected_surfaces), parcel in zip(expected_parcels, parcel_mapping.parcels):
        assert parcel.name == name
        assert len(parcel.vertices) == 2
        for vertices, orientation, (length, first_element, last_element) in zip(parcel.vertices, ('LEFT', 'RIGHT'), expected_surfaces):
            assert len(vertices) == length
            assert vertices[0] == first_element
            assert vertices[-1] == last_element
            assert vertices.brain_structure == f'CIFTI_STRUCTURE_CORTEX_{orientation}'