import os
import os.path as op
import datetime
import string
import networkx as nx
from ...utils.filemanip import split_filename
from ..base import (
from .base import CFFBaseInterface, have_cfflib
class CFFConverterInputSpec(BaseInterfaceInputSpec):
    graphml_networks = InputMultiPath(File(exists=True), desc='list of graphML networks')
    gpickled_networks = InputMultiPath(File(exists=True), desc='list of gpickled Networkx graphs')
    gifti_surfaces = InputMultiPath(File(exists=True), desc='list of GIFTI surfaces')
    gifti_labels = InputMultiPath(File(exists=True), desc='list of GIFTI labels')
    nifti_volumes = InputMultiPath(File(exists=True), desc='list of NIFTI volumes')
    tract_files = InputMultiPath(File(exists=True), desc='list of Trackvis fiber files')
    timeseries_files = InputMultiPath(File(exists=True), desc='list of HDF5 timeseries files')
    script_files = InputMultiPath(File(exists=True), desc='list of script files to include')
    data_files = InputMultiPath(File(exists=True), desc='list of external data files (i.e. Numpy, HD5, XML) ')
    title = traits.Str(desc='Connectome Title')
    creator = traits.Str(desc='Creator')
    email = traits.Str(desc='Email address')
    publisher = traits.Str(desc='Publisher')
    license = traits.Str(desc='License')
    rights = traits.Str(desc='Rights')
    references = traits.Str(desc='References')
    relation = traits.Str(desc='Relation')
    species = traits.Str('Homo sapiens', desc='Species', usedefault=True)
    description = traits.Str('Created with the Nipype CFF converter', desc='Description', usedefault=True)
    out_file = File('connectome.cff', usedefault=True, desc='Output connectome file')