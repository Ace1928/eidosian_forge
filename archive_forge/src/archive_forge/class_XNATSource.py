import glob
import fnmatch
import string
import json
import os
import os.path as op
import shutil
import subprocess
import re
import copy
import tempfile
from os.path import join, dirname
from warnings import warn
from .. import config, logging
from ..utils.filemanip import (
from ..utils.misc import human_order_sorted, str2bool
from .base import (
class XNATSource(LibraryBaseInterface, IOBase):
    """
    Pull data from an XNAT server.

    Generic XNATSource module that wraps around the pyxnat module in
    an intelligent way for neuroimaging tasks to grab files and data
    from an XNAT server.

    Examples
    --------
    Pick all files from current directory

    >>> dg = XNATSource()
    >>> dg.inputs.template = '*'

    >>> dg = XNATSource(infields=['project','subject','experiment','assessor','inout'])
    >>> dg.inputs.query_template = '/projects/%s/subjects/%s/experiments/%s'                '/assessors/%s/%s_resources/files'
    >>> dg.inputs.project = 'IMAGEN'
    >>> dg.inputs.subject = 'IMAGEN_000000001274'
    >>> dg.inputs.experiment = '*SessionA*'
    >>> dg.inputs.assessor = '*ADNI_MPRAGE_nii'
    >>> dg.inputs.inout = 'out'

    >>> dg = XNATSource(infields=['sid'],outfields=['struct','func'])
    >>> dg.inputs.query_template = '/projects/IMAGEN/subjects/%s/experiments/*SessionA*'                '/assessors/*%s_nii/out_resources/files'
    >>> dg.inputs.query_template_args['struct'] = [['sid','ADNI_MPRAGE']]
    >>> dg.inputs.query_template_args['func'] = [['sid','EPI_faces']]
    >>> dg.inputs.sid = 'IMAGEN_000000001274'

    """
    input_spec = XNATSourceInputSpec
    output_spec = DynamicTraitedSpec
    _pkg = 'pyxnat'

    def __init__(self, infields=None, outfields=None, **kwargs):
        """
        Parameters
        ----------
        infields : list of str
            Indicates the input fields to be dynamically created

        outfields: list of str
            Indicates output fields to be dynamically created

        See class examples for usage

        """
        super(XNATSource, self).__init__(**kwargs)
        undefined_traits = {}
        self._infields = infields
        if infields:
            for key in infields:
                self.inputs.add_trait(key, traits.Any)
                undefined_traits[key] = Undefined
            self.inputs.query_template_args['outfiles'] = [infields]
        if outfields:
            self.inputs.add_trait('field_template', traits.Dict(traits.Enum(outfields), desc='arguments that fit into query_template'))
            undefined_traits['field_template'] = Undefined
            outdict = {}
            for key in outfields:
                outdict[key] = []
            self.inputs.query_template_args = outdict
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)

    def _add_output_traits(self, base):
        """

        Using traits.Any instead out OutputMultiPath till add_trait bug
        is fixed.
        """
        return add_traits(base, list(self.inputs.query_template_args.keys()))

    def _list_outputs(self):
        import pyxnat
        cache_dir = self.inputs.cache_dir or tempfile.gettempdir()
        if self.inputs.config:
            xnat = pyxnat.Interface(config=self.inputs.config)
        else:
            xnat = pyxnat.Interface(self.inputs.server, self.inputs.user, self.inputs.pwd, cache_dir)
        if self._infields:
            for key in self._infields:
                value = getattr(self.inputs, key)
                if not isdefined(value):
                    msg = "%s requires a value for input '%s' because it was listed in 'infields'" % (self.__class__.__name__, key)
                    raise ValueError(msg)
        outputs = {}
        for key, args in list(self.inputs.query_template_args.items()):
            outputs[key] = []
            template = self.inputs.query_template
            if hasattr(self.inputs, 'field_template') and isdefined(self.inputs.field_template) and (key in self.inputs.field_template):
                template = self.inputs.field_template[key]
            if not args:
                file_objects = xnat.select(template).get('obj')
                if file_objects == []:
                    raise IOError('Template %s returned no files' % template)
                outputs[key] = simplify_list([str(file_object.get()) for file_object in file_objects if file_object.exists()])
            for argnum, arglist in enumerate(args):
                maxlen = 1
                for arg in arglist:
                    if isinstance(arg, (str, bytes)) and hasattr(self.inputs, arg):
                        arg = getattr(self.inputs, arg)
                    if isinstance(arg, list):
                        if maxlen > 1 and len(arg) != maxlen:
                            raise ValueError('incompatible number of arguments for %s' % key)
                        if len(arg) > maxlen:
                            maxlen = len(arg)
                outfiles = []
                for i in range(maxlen):
                    argtuple = []
                    for arg in arglist:
                        if isinstance(arg, (str, bytes)) and hasattr(self.inputs, arg):
                            arg = getattr(self.inputs, arg)
                        if isinstance(arg, list):
                            argtuple.append(arg[i])
                        else:
                            argtuple.append(arg)
                    if argtuple:
                        target = template % tuple(argtuple)
                        file_objects = xnat.select(target).get('obj')
                        if file_objects == []:
                            raise IOError('Template %s returned no files' % target)
                        outfiles = simplify_list([str(file_object.get()) for file_object in file_objects if file_object.exists()])
                    else:
                        file_objects = xnat.select(template).get('obj')
                        if file_objects == []:
                            raise IOError('Template %s returned no files' % template)
                        outfiles = simplify_list([str(file_object.get()) for file_object in file_objects if file_object.exists()])
                    outputs[key].insert(i, outfiles)
            if len(outputs[key]) == 0:
                outputs[key] = None
            elif len(outputs[key]) == 1:
                outputs[key] = outputs[key][0]
        return outputs