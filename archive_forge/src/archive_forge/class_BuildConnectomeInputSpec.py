import os
import os.path as op
from ..base import CommandLineInputSpec, traits, TraitedSpec, File, isdefined
from .base import MRTrix3Base
class BuildConnectomeInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3, desc='input tractography')
    in_parc = File(exists=True, argstr='%s', position=-2, desc='parcellation file')
    out_file = File('connectome.csv', argstr='%s', mandatory=True, position=-1, usedefault=True, desc='output file after processing')
    nthreads = traits.Int(argstr='-nthreads %d', desc='number of threads. if zero, the number of available cpus will be used', nohash=True)
    vox_lookup = traits.Bool(argstr='-assignment_voxel_lookup', desc='use a simple voxel lookup value at each streamline endpoint')
    search_radius = traits.Float(argstr='-assignment_radial_search %f', desc='perform a radial search from each streamline endpoint to locate the nearest node. Argument is the maximum radius in mm; if no node is found within this radius, the streamline endpoint is not assigned to any node.')
    search_reverse = traits.Float(argstr='-assignment_reverse_search %f', desc='traverse from each streamline endpoint inwards along the streamline, in search of the last node traversed by the streamline. Argument is the maximum traversal length in mm (set to 0 to allow search to continue to the streamline midpoint).')
    search_forward = traits.Float(argstr='-assignment_forward_search %f', desc='project the streamline forwards from the endpoint in search of aparcellation node voxel. Argument is the maximum traversal length in mm.')
    metric = traits.Enum('count', 'meanlength', 'invlength', 'invnodevolume', 'mean_scalar', 'invlength_invnodevolume', argstr='-metric %s', desc='specify the edge weight metric')
    in_scalar = File(exists=True, argstr='-image %s', desc='provide the associated image for the mean_scalar metric')
    in_weights = File(exists=True, argstr='-tck_weights_in %s', desc='specify a text scalar file containing the streamline weights')
    keep_unassigned = traits.Bool(argstr='-keep_unassigned', desc='By default, the program discards the information regarding those streamlines that are not successfully assigned to a node pair. Set this option to keep these values (will be the first row/column in the output matrix)')
    zero_diagonal = traits.Bool(argstr='-zero_diagonal', desc='set all diagonal entries in the matrix to zero (these represent streamlines that connect to the same node at both ends)')