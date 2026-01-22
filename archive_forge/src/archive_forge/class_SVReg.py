import os
import re as regex
from ..base import (
class SVReg(CommandLine):
    """
    surface and volume registration (svreg)
    This program registers a subject's BrainSuite-processed volume and surfaces
    to an atlas, allowing for automatic labelling of volume and surface ROIs.

    For more information, please see:
    http://brainsuite.org/processing/svreg/usage/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> svreg = brainsuite.SVReg()
    >>> svreg.inputs.subjectFilePrefix = 'home/user/btestsubject/testsubject'
    >>> svreg.inputs.refineOutputs = True
    >>> svreg.inputs.skipToVolumeReg = False
    >>> svreg.inputs. keepIntermediates = True
    >>> svreg.inputs.verbosity2 = True
    >>> svreg.inputs.displayTimestamps = True
    >>> svreg.inputs.useSingleThreading = True
    >>> results = svreg.run() #doctest: +SKIP


    """
    input_spec = SVRegInputSpec
    _cmd = 'svreg.sh'

    def _format_arg(self, name, spec, value):
        if name == 'subjectFilePrefix' or name == 'atlasFilePrefix' or name == 'curveMatchingInstructions':
            return spec.argstr % os.path.expanduser(value)
        if name == 'dataSinkDelay':
            return spec.argstr % ''
        return super(SVReg, self)._format_arg(name, spec, value)