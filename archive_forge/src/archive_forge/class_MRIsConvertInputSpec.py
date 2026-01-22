import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class MRIsConvertInputSpec(FSTraitedSpec):
    """
    Uses Freesurfer's mris_convert to convert surface files to various formats
    """
    annot_file = File(exists=True, argstr='--annot %s', desc='input is annotation or gifti label data')
    parcstats_file = File(exists=True, argstr='--parcstats %s', desc='infile is name of text file containing label/val pairs')
    label_file = File(exists=True, argstr='--label %s', desc='infile is .label file, label is name of this label')
    scalarcurv_file = File(exists=True, argstr='-c %s', desc='input is scalar curv overlay file (must still specify surface)')
    functional_file = File(exists=True, argstr='-f %s', desc='input is functional time-series or other multi-frame data (must specify surface)')
    labelstats_outfile = File(exists=False, argstr='--labelstats %s', desc='outfile is name of gifti file to which label stats will be written')
    patch = traits.Bool(argstr='-p', desc='input is a patch, not a full surface')
    rescale = traits.Bool(argstr='-r', desc='rescale vertex xyz so total area is same as group average')
    normal = traits.Bool(argstr='-n', desc='output is an ascii file where vertex data')
    xyz_ascii = traits.Bool(argstr='-a', desc='Print only surface xyz to ascii file')
    vertex = traits.Bool(argstr='-v', desc='Writes out neighbors of a vertex in each row')
    scale = traits.Float(argstr='-s %.3f', desc='scale vertex xyz by scale')
    dataarray_num = traits.Int(argstr='--da_num %d', desc="if input is gifti, 'num' specifies which data array to use")
    talairachxfm_subjid = traits.String(argstr='-t %s', desc='apply talairach xfm of subject to vertex xyz')
    origname = traits.String(argstr='-o %s', desc='read orig positions')
    in_file = File(exists=True, mandatory=True, position=-2, argstr='%s', desc='File to read/convert')
    out_file = File(argstr='%s', position=-1, genfile=True, xor=['out_datatype'], mandatory=True, desc='output filename or True to generate one')
    out_datatype = traits.Enum('asc', 'ico', 'tri', 'stl', 'vtk', 'gii', 'mgh', 'mgz', xor=['out_file'], mandatory=True, desc="These file formats are supported:  ASCII:       .ascICO: .ico, .tri GEO: .geo STL: .stl VTK: .vtk GIFTI: .gii MGH surface-encoded 'volume': .mgh, .mgz")
    to_scanner = traits.Bool(argstr='--to-scanner', desc='convert coordinates from native FS (tkr) coords to scanner coords')
    to_tkr = traits.Bool(argstr='--to-tkr', desc='convert coordinates from scanner coords to native FS (tkr) coords')