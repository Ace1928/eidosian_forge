import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dscalar():
    img = nib.load(os.path.join(test_directory, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii'))
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.ScalarAxis)
    assert len(axes[0]) == 2
    assert axes[0].name[0] == 'MyelinMap_BC_decurv'
    assert axes[0].name[1] == 'corrThickness'
    assert axes[0].meta[0] == {'PaletteColorMapping': '<PaletteColorMapping Version="1">\n   <ScaleMode>MODE_AUTO_SCALE_PERCENTAGE</ScaleMode>\n   <AutoScalePercentageValues>98.000000 2.000000 2.000000 98.000000</AutoScalePercentageValues>\n   <UserScaleValues>-100.000000 0.000000 0.000000 100.000000</UserScaleValues>\n   <PaletteName>ROY-BIG-BL</PaletteName>\n   <InterpolatePalette>true</InterpolatePalette>\n   <DisplayPositiveData>true</DisplayPositiveData>\n   <DisplayZeroData>false</DisplayZeroData>\n   <DisplayNegativeData>true</DisplayNegativeData>\n   <ThresholdTest>THRESHOLD_TEST_SHOW_OUTSIDE</ThresholdTest>\n   <ThresholdType>THRESHOLD_TYPE_OFF</ThresholdType>\n   <ThresholdFailureInGreen>false</ThresholdFailureInGreen>\n   <ThresholdNormalValues>-1.000000 1.000000</ThresholdNormalValues>\n   <ThresholdMappedValues>-1.000000 1.000000</ThresholdMappedValues>\n   <ThresholdMappedAvgAreaValues>-1.000000 1.000000</ThresholdMappedAvgAreaValues>\n   <ThresholdDataName></ThresholdDataName>\n   <ThresholdRangeMode>PALETTE_THRESHOLD_RANGE_MODE_MAP</ThresholdRangeMode>\n</PaletteColorMapping>'}
    check_Conte69(axes[1])
    check_rewrite(arr, axes)