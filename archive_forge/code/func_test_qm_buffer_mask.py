import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.qmmm import ForceQMMM, RescaledCalculator
from ase.eos import EquationOfState
from ase.optimize import FIRE
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
@pytest.mark.slow
def test_qm_buffer_mask(qm_calc, mm_calc, bulk_at):
    """
    test number of atoms in qm_buffer_mask for
    spherical region in a fully periodic cell
    also tests that "region" array returns the same mapping
    """
    alat = bulk_at.cell[0, 0]
    N_cell_geom = 10
    at0 = bulk_at * N_cell_geom
    r = at0.get_distances(0, np.arange(len(at0)), mic=True)
    print('N_cell', N_cell_geom, 'N_MM', len(at0), 'Size', N_cell_geom * alat)
    qm_rc = 5.37
    for R_QM in [0.001, alat / np.sqrt(2.0) + 0.001, alat + 0.001]:
        at = at0.copy()
        qm_mask = r < R_QM
        qm_buffer_mask_ref = r < 2 * qm_rc + R_QM
        _, r_qm_buffer = get_distances(at.positions[qm_buffer_mask_ref], at.positions[qm_mask], at.cell, at.pbc)
        updated_qm_buffer_mask = np.ones_like(at[qm_buffer_mask_ref])
        for i, r_qm in enumerate(r_qm_buffer):
            if r_qm.min() > 2 * qm_rc:
                updated_qm_buffer_mask[i] = False
        qm_buffer_mask_ref[qm_buffer_mask_ref] = updated_qm_buffer_mask
        "\n        print(f'R_QM             {R_QM}   N_QM        {qm_mask.sum()}')\n        print(f'R_QM + buffer: {2 * qm_rc + R_QM:.2f}'\n              f' N_QM_buffer {qm_buffer_mask_ref.sum()}')\n        print(f'                     N_total:    {len(at)}')\n        "
        qmmm = ForceQMMM(at, qm_mask, qm_calc, mm_calc, buffer_width=2 * qm_rc)
        qmmm.initialize_qm_buffer_mask(at)
        assert qmmm.qm_buffer_mask.sum() == qm_buffer_mask_ref.sum()
        qm_cluster = qmmm.get_qm_cluster(at)
        assert len(qm_cluster) == qm_buffer_mask_ref.sum()
        region = qmmm.get_region_from_masks(at)
        qm_mask_region = region == 'QM'
        assert qm_mask_region.sum() == qm_mask.sum()
        buffer_mask_region = region == 'buffer'
        assert qm_mask_region.sum() + buffer_mask_region.sum() == qm_buffer_mask_ref.sum()