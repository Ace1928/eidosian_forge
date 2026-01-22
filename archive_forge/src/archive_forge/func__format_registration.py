import os
from ...utils.filemanip import ensure_list
from ..base import TraitedSpec, File, Str, traits, InputMultiPath, isdefined
from .base import ANTSCommand, ANTSCommandInputSpec, LOCAL_DEFAULT_NUMBER_OF_THREADS
def _format_registration(self):
    retval = []
    for ii in range(len(self.inputs.transforms)):
        retval.append('--transform %s' % self._format_transform(ii))
        for metric in self._format_metric(ii):
            retval.append('--metric %s' % metric)
        retval.append('--convergence %s' % self._format_convergence(ii))
        if isdefined(self.inputs.sigma_units):
            retval.append('--smoothing-sigmas %s%s' % (self._format_xarray(self.inputs.smoothing_sigmas[ii]), self.inputs.sigma_units[ii]))
        else:
            retval.append('--smoothing-sigmas %s' % self._format_xarray(self.inputs.smoothing_sigmas[ii]))
        retval.append('--shrink-factors %s' % self._format_xarray(self.inputs.shrink_factors[ii]))
        if isdefined(self.inputs.use_estimate_learning_rate_once):
            retval.append('--use-estimate-learning-rate-once %d' % self.inputs.use_estimate_learning_rate_once[ii])
        if isdefined(self.inputs.use_histogram_matching):
            if isinstance(self.inputs.use_histogram_matching, bool):
                histval = self.inputs.use_histogram_matching
            else:
                histval = self.inputs.use_histogram_matching[ii]
            retval.append('--use-histogram-matching %d' % histval)
        if isdefined(self.inputs.restrict_deformation):
            retval.append('--restrict-deformation %s' % self._format_xarray(self.inputs.restrict_deformation[ii]))
        if any((isdefined(self.inputs.fixed_image_masks), isdefined(self.inputs.moving_image_masks))):
            if isdefined(self.inputs.fixed_image_masks):
                fixed_masks = ensure_list(self.inputs.fixed_image_masks)
                fixed_mask = fixed_masks[ii if len(fixed_masks) > 1 else 0]
            else:
                fixed_mask = 'NULL'
            if isdefined(self.inputs.moving_image_masks):
                moving_masks = ensure_list(self.inputs.moving_image_masks)
                moving_mask = moving_masks[ii if len(moving_masks) > 1 else 0]
            else:
                moving_mask = 'NULL'
            retval.append('--masks [ %s, %s ]' % (fixed_mask, moving_mask))
    return ' '.join(retval)