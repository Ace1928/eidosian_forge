import os
from ...utils.filemanip import split_filename
from ..base import (
class DTMetricInputSpec(CommandLineInputSpec):
    eigen_data = File(exists=True, argstr='-inputfile %s', mandatory=True, desc='voxel-order data filename')
    metric = traits.Enum('fa', 'md', 'rd', 'l1', 'l2', 'l3', 'tr', 'ra', '2dfa', 'cl', 'cp', 'cs', argstr='-stat %s', mandatory=True, desc='Specifies the metric to compute.')
    inputdatatype = traits.Enum('double', 'float', 'long', 'int', 'short', 'char', argstr='-inputdatatype %s', usedefault=True, desc='Specifies the data type of the input data.')
    outputdatatype = traits.Enum('double', 'float', 'long', 'int', 'short', 'char', argstr='-outputdatatype %s', usedefault=True, desc='Specifies the data type of the output data.')
    data_header = File(argstr='-header %s', exists=True, desc='A Nifti .nii or .nii.gz file containing the header information. Usually this will be the header of the raw data file from which the diffusion tensors were reconstructed.')
    outputfile = File(argstr='-outputfile %s', genfile=True, desc='Output name. Output will be a .nii.gz file if data_header is provided andin voxel order with outputdatatype datatype (default: double) otherwise.')