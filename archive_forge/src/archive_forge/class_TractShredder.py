import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class TractShredder(StdOutCommandLine):
    """
    Extracts bunches of streamlines.

    tractshredder works in a similar way to shredder, but processes streamlines instead of scalar data.
    The input is raw streamlines, in the format produced by track or procstreamlines.

    The program first makes an initial offset of offset tracts.  It then reads and outputs a group of
    bunchsize tracts, skips space tracts, and repeats until there is no more input.

    Examples
    --------

    >>> import nipype.interfaces.camino as cmon
    >>> shred = cmon.TractShredder()
    >>> shred.inputs.in_file = 'tract_data.Bfloat'
    >>> shred.inputs.offset = 0
    >>> shred.inputs.bunchsize = 1
    >>> shred.inputs.space = 2
    >>> shred.run()                  # doctest: +SKIP
    """
    _cmd = 'tractshredder'
    input_spec = TractShredderInputSpec
    output_spec = TractShredderOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['shredded'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '_shredded'