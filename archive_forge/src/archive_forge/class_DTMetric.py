import os
from ...utils.filemanip import split_filename
from ..base import (
class DTMetric(CommandLine):
    """
    Computes tensor metric statistics based on the eigenvalues l1 >= l2 >= l3
    typically obtained from ComputeEigensystem.

    The full list of statistics is:

     - <cl> = (l1 - l2) / l1 , a measure of linearity
     - <cp> = (l2 - l3) / l1 , a measure of planarity
     - <cs> = l3 / l1 , a measure of isotropy
       with: cl + cp + cs = 1
     - <l1> = first eigenvalue
     - <l2> = second eigenvalue
     - <l3> = third eigenvalue
     - <tr> = l1 + l2 + l3
     - <md> = tr / 3
     - <rd> = (l2 + l3) / 2
     - <fa> = fractional anisotropy. (Basser et al, J Magn Reson B 1996)
     - <ra> = relative anisotropy (Basser et al, J Magn Reson B 1996)
     - <2dfa> = 2D FA of the two minor eigenvalues l2 and l3
       i.e. sqrt( 2 * [(l2 - <l>)^2 + (l3 - <l>)^2] / (l2^2 + l3^2) )
       with: <l> = (l2 + l3) / 2


    Example
    -------
    Compute the CP planar metric as float data type.

    >>> import nipype.interfaces.camino as cam
    >>> dtmetric = cam.DTMetric()
    >>> dtmetric.inputs.eigen_data = 'dteig.Bdouble'
    >>> dtmetric.inputs.metric = 'cp'
    >>> dtmetric.inputs.outputdatatype = 'float'
    >>> dtmetric.run()                  # doctest: +SKIP

    """
    _cmd = 'dtshape'
    input_spec = DTMetricInputSpec
    output_spec = DTMetricOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['metric_stats'] = os.path.abspath(self._gen_outfilename())
        return outputs

    def _gen_outfilename(self):
        return self._gen_outputfile()

    def _gen_outputfile(self):
        outputfile = self.inputs.outputfile
        if not isdefined(outputfile):
            outputfile = self._gen_filename('outputfile')
        return outputfile

    def _gen_filename(self, name):
        if name == 'outputfile':
            _, name, _ = split_filename(self.inputs.eigen_data)
            metric = self.inputs.metric
            datatype = self.inputs.outputdatatype
            if isdefined(self.inputs.data_header):
                filename = name + '_' + metric + '.nii.gz'
            else:
                filename = name + '_' + metric + '.B' + datatype
        return filename