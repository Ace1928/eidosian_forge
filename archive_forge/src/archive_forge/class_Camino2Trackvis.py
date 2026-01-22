import os
from ...utils.filemanip import split_filename
from ..base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
class Camino2Trackvis(CommandLine):
    """Wraps camino_to_trackvis from Camino-Trackvis

    Convert files from camino .Bfloat format to trackvis .trk format.

    Example
    -------

    >>> import nipype.interfaces.camino2trackvis as cam2trk
    >>> c2t = cam2trk.Camino2Trackvis()
    >>> c2t.inputs.in_file = 'data.Bfloat'
    >>> c2t.inputs.out_file = 'streamlines.trk'
    >>> c2t.inputs.min_length = 30
    >>> c2t.inputs.data_dims = [128, 104, 64]
    >>> c2t.inputs.voxel_dims = [2.0, 2.0, 2.0]
    >>> c2t.inputs.voxel_order = 'LAS'
    >>> c2t.run()                  # doctest: +SKIP
    """
    _cmd = 'camino_to_trackvis'
    input_spec = Camino2TrackvisInputSpec
    output_spec = Camino2TrackvisOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['trackvis'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '.trk'