import os
from pathlib import Path
from nipype.interfaces.base import (
from nipype.interfaces.cat12.base import Cell
from nipype.interfaces.spm import SPMCommand
from nipype.interfaces.spm.base import (
from nipype.utils.filemanip import split_filename, fname_presuffix
class CAT12Segment(SPMCommand):
    """
    CAT12: Segmentation

    This toolbox is an extension to the default segmentation in SPM12, but uses a completely different segmentation
    approach.
    The segmentation approach is based on an Adaptive Maximum A Posterior (MAP) technique without the need for a priori
    information about tissue probabilities. That is, the Tissue Probability Maps (TPM) are not used constantly in the
    sense of the classical Unified Segmentation approach (Ashburner et. al. 2005), but just for spatial normalization.
    The following AMAP estimation is adaptive in the sense that local variations of the parameters (i.e., means and
    variance) are modeled as slowly varying spatial functions (Rajapakse et al. 1997). This not only accounts for
    intensity inhomogeneities but also for other local variations of intensity.
    Additionally, the segmentation approach uses a Partial Volume Estimation (PVE) with a simplified mixed model of at
    most two tissue types (Tohka et al. 2004). We start with an initial segmentation into three pure classes: gray
    matter (GM), white matter (WM), and cerebrospinal fluid (CSF) based on the above described AMAP estimation. The
    initial segmentation is followed by a PVE of two additional mixed classes: GM-WM and GM-CSF. This results in an
    estimation of the amount (or fraction) of each pure tissue type present in every voxel (as single voxels - given by
    Another important extension to the SPM12 segmentation is the integration of the Dartel or Geodesic Shooting
    registration into the toolbox by an already existing Dartel/Shooting template in MNI space. This template was
    derived from 555 healthy control subjects of the IXI-database (http://www.brain-development.org) and provides the
    several Dartel or Shooting iterations. Thus, for the majority of studies the creation of sample-specific templates
    is not necessary anymore and is mainly recommended for children data.'};

    http://www.neuro.uni-jena.de/cat12/CAT12-Manual.pdf#page=15

    Examples
    --------
    >>> path_mr = 'structural.nii'
    >>> cat = CAT12Segment(in_files=path_mr)
    >>> cat.run() # doctest: +SKIP
    """
    input_spec = CAT12SegmentInputSpec
    output_spec = CAT12SegmentOutputSpec

    def __init__(self, **inputs):
        _local_version = SPMCommand().version
        if _local_version and '12.' in _local_version:
            self._jobtype = 'tools'
            self._jobname = 'cat.estwrite'
        SPMCommand.__init__(self, **inputs)

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == 'in_files':
            if isinstance(val, list):
                return scans_for_fnames(val)
            else:
                return scans_for_fname(val)
        elif opt in ['tpm', 'shooting_tpm']:
            return Cell2Str(val)
        return super(CAT12Segment, self)._format_arg(opt, spec, val)

    def _list_outputs(self):
        outputs = self._outputs().get()
        f = self.inputs.in_files[0]
        pth, base, ext = split_filename(f)
        outputs['mri_images'] = [str(mri) for mri in Path(pth).glob('mri/*') if mri.is_file()]
        for tidx, tissue in enumerate(['gm', 'wm', 'csf']):
            for image, prefix in [('modulated', 'mw'), ('dartel', 'r'), ('native', '')]:
                outtype = f'{tissue}_output_{image}'
                if isdefined(getattr(self.inputs, outtype)) and getattr(self.inputs, outtype):
                    outfield = f'{tissue}_{image}_image'
                    prefix = os.path.join('mri', f'{prefix}p{tidx + 1}')
                    if image != 'dartel':
                        outputs[outfield] = fname_presuffix(f, prefix=prefix)
                    else:
                        outputs[outfield] = fname_presuffix(f, prefix=prefix, suffix='_rigid')
        if self.inputs.save_bias_corrected:
            outputs['bias_corrected_image'] = fname_presuffix(f, prefix=os.path.join('mri', 'wmi'))
        outputs['surface_files'] = [str(surf) for surf in Path(pth).glob('surf/*') if surf.is_file()]
        for hemisphere in ['rh', 'lh']:
            for suffix in ['central', 'sphere']:
                outfield = f'{hemisphere}_{suffix}_surface'
                outputs[outfield] = fname_presuffix(f, prefix=os.path.join('surf', f'{hemisphere}.{suffix}.'), suffix='.gii', use_ext=False)
        outputs['report_files'] = outputs['report_files'] = [str(report) for report in Path(pth).glob('report/*') if report.is_file()]
        outputs['report'] = fname_presuffix(f, prefix=os.path.join('report', f'cat_'), suffix='.xml', use_ext=False)
        outputs['label_files'] = [str(label) for label in Path(pth).glob('label/*') if label.is_file()]
        outputs['label_rois'] = fname_presuffix(f, prefix=os.path.join('label', 'catROIs_'), suffix='.xml', use_ext=False)
        outputs['label_roi'] = fname_presuffix(f, prefix=os.path.join('label', 'catROI_'), suffix='.xml', use_ext=False)
        return outputs