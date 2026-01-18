from copy import deepcopy
import numpy as np
import pytest
import nibabel.cifti2.cifti2_axes as axes
from .test_cifti2io_axes import check_rewrite
def test_parcels():
    """
    Test the introspection and creation of CIFTI-2 Parcel axes
    """
    prc = get_parcels()
    assert isinstance(prc, axes.ParcelsAxis)
    assert prc[0] == ('mixed',) + prc['mixed']
    assert prc['mixed'][0].shape == (3, 3)
    assert len(prc['mixed'][1]) == 1
    assert prc['mixed'][1]['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (3,)
    assert prc[1] == ('volume',) + prc['volume']
    assert prc['volume'][0].shape == (4, 3)
    assert len(prc['volume'][1]) == 0
    assert prc[2] == ('surface',) + prc['surface']
    assert prc['surface'][0].shape == (0, 3)
    assert len(prc['surface'][1]) == 1
    assert prc['surface'][1]['CIFTI_STRUCTURE_OTHER'].shape == (4,)
    prc2 = prc + prc
    assert len(prc2) == 6
    assert (prc2.affine == prc.affine).all()
    assert prc2.nvertices == prc.nvertices
    assert prc2.volume_shape == prc.volume_shape
    assert prc2[:3] == prc
    assert prc2[3:] == prc
    assert prc2[3:]['mixed'][0].shape == (3, 3)
    assert len(prc2[3:]['mixed'][1]) == 1
    assert prc2[3:]['mixed'][1]['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (3,)
    with pytest.raises(IndexError):
        prc['non_existent']
    prc['surface']
    with pytest.raises(IndexError):
        prc2['surface']
    prc.affine = np.eye(4)
    with pytest.raises(ValueError):
        prc.affine = np.eye(3)
    with pytest.raises(ValueError):
        prc.affine = np.eye(4).flatten()
    prc.volume_shape = (5, 3, 1)
    with pytest.raises(ValueError):
        prc.volume_shape = (5.0, 3, 1)
    with pytest.raises(ValueError):
        prc.volume_shape = (5, 3, 1, 4)
    with pytest.raises(Exception):
        prc + get_label()
    prc = get_parcels()
    other_prc = get_parcels()
    prc + other_prc
    other_prc = get_parcels()
    other_prc.affine = np.eye(4) * 2
    with pytest.raises(ValueError):
        prc + other_prc
    other_prc = get_parcels()
    other_prc.volume_shape = (20, 3, 4)
    with pytest.raises(ValueError):
        prc + other_prc
    prc = get_parcels()
    assert prc != get_scalar()
    prc_other = deepcopy(prc)
    assert prc == prc_other
    assert prc != prc_other[:2]
    assert prc == prc_other[:]
    prc_other.affine[0, 0] = 10
    assert prc != prc_other
    prc_other = deepcopy(prc)
    prc_other.affine = None
    assert prc != prc_other
    assert prc_other != prc
    assert (prc + prc_other).affine is not None
    assert (prc_other + prc).affine is not None
    prc_other = deepcopy(prc)
    prc_other.volume_shape = (10, 3, 4)
    assert prc != prc_other
    with pytest.raises(ValueError):
        prc + prc_other
    prc_other = deepcopy(prc)
    prc_other.nvertices['CIFTI_STRUCTURE_CORTEX_LEFT'] = 80
    assert prc != prc_other
    with pytest.raises(ValueError):
        prc + prc_other
    prc_other = deepcopy(prc)
    prc_other.voxels[0] = np.ones((2, 3), dtype='i4')
    assert prc != prc_other
    prc_other = deepcopy(prc)
    prc_other.voxels[0] = prc_other.voxels * 2
    assert prc != prc_other
    prc_other = deepcopy(prc)
    prc_other.vertices[0]['CIFTI_STRUCTURE_CORTEX_LEFT'] = np.ones((8,), dtype='i4')
    assert prc != prc_other
    prc_other = deepcopy(prc)
    prc_other.vertices[0]['CIFTI_STRUCTURE_CORTEX_LEFT'] *= 2
    assert prc != prc_other
    prc_other = deepcopy(prc)
    prc_other.name[0] = 'new_name'
    assert prc != prc_other
    test_parcel = axes.ParcelsAxis(voxels=[np.ones((3, 2), dtype=int)], vertices=[{}], name=['single_voxel'], affine=np.eye(4), volume_shape=(2, 3, 4))
    assert len(test_parcel) == 1
    test_parcel = axes.ParcelsAxis(voxels=[np.ones((3, 2), dtype=int), np.zeros((3, 2), dtype=int)], vertices=[{}, {}], name=['first_parcel', 'second_parcel'], affine=np.eye(4), volume_shape=(2, 3, 4))
    assert len(test_parcel) == 2
    test_parcel = axes.ParcelsAxis(voxels=[np.ones((3, 2), dtype=int), np.zeros((5, 2), dtype=int)], vertices=[{}, {}], name=['first_parcel', 'second_parcel'], affine=np.eye(4), volume_shape=(2, 3, 4))
    assert len(test_parcel) == 2
    with pytest.raises(ValueError):
        axes.ParcelsAxis(voxels=[np.ones((3, 2), dtype=int)], vertices=[{}], name=[['single_voxel']], affine=np.eye(4), volume_shape=(2, 3, 4))