import os
from ... import logging
from ...utils.filemanip import fname_presuffix
from ..base import CommandLine
from nipype.interfaces.fsl.base import Info
import warnings
class DTITKRenameMixin(object):

    def __init__(self, *args, **kwargs):
        classes = [cls.__name__ for cls in self.__class__.mro()]
        dep_name = classes[0]
        rename_idx = classes.index('DTITKRenameMixin')
        new_name = classes[rename_idx + 1]
        warnings.warn('The {} interface has been renamed to {}\nPlease see the documentation for DTI-TK interfaces, as some inputs have been added or renamed for clarity.'.format(dep_name, new_name), DeprecationWarning)
        super(DTITKRenameMixin, self).__init__(*args, **kwargs)