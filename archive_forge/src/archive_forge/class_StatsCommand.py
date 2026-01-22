import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class StatsCommand(NiftySegCommand):
    """
    Base Command Interface for seg_stats interfaces.

    The executable seg_stats enables the estimation of image statistics on
    continuous voxel intensities (average, standard deviation, min/max, robust
    range, percentiles, sum, probabilistic volume, entropy, etc) either over
    the full image or on a per slice basis (slice axis can be specified),
    statistics over voxel coordinates (location of max, min and centre of
    mass, bounding box, etc) and statistics over categorical images (e.g. per
    region volume, count, average, Dice scores, etc). These statistics are
    robust to the presence of NaNs, and can be constrained by a mask and/or
    thresholded at a certain level.
    """
    _cmd = get_custom_path('seg_stats', env_dir='NIFTYSEGDIR')
    input_spec = StatsInput
    output_spec = StatsOutput

    def _parse_stdout(self, stdout):
        out = []
        for string_line in stdout.split('\n'):
            if string_line.startswith('#'):
                continue
            if len(string_line) <= 1:
                continue
            line = [float(s) for s in string_line.split()]
            out.append(line)
        return np.array(out).squeeze()

    def _run_interface(self, runtime):
        new_runtime = super(StatsCommand, self)._run_interface(runtime)
        self.output = self._parse_stdout(new_runtime.stdout)
        return new_runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output'] = self.output
        return outputs