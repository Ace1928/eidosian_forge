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
class BIDSDataGrabber(LibraryBaseInterface, IOBase):
    """BIDS datagrabber module that wraps around pybids to allow arbitrary
    querying of BIDS datasets.

    Examples
    --------

    .. setup::

        >>> try:
        ...     import bids
        ... except ImportError:
        ...     pytest.skip()

    By default, the BIDSDataGrabber fetches anatomical and functional images
    from a project, and makes BIDS entities (e.g. subject) available for
    filtering outputs.

    >>> bg = BIDSDataGrabber()
    >>> bg.inputs.base_dir = 'ds005/'
    >>> bg.inputs.subject = '01'
    >>> results = bg.run() # doctest: +SKIP


    Dynamically created, user-defined output fields can also be defined to
    return different types of outputs from the same project. All outputs
    are filtered on common entities, which can be explicitly defined as
    infields.

    >>> bg = BIDSDataGrabber(infields = ['subject'])
    >>> bg.inputs.base_dir = 'ds005/'
    >>> bg.inputs.subject = '01'
    >>> bg.inputs.output_query['dwi'] = dict(datatype='dwi')
    >>> results = bg.run() # doctest: +SKIP

    """
    input_spec = BIDSDataGrabberInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True
    _pkg = 'bids'

    def __init__(self, infields=None, **kwargs):
        """
        Parameters
        ----------
        infields : list of str
            Indicates the input fields to be dynamically created
        """
        super(BIDSDataGrabber, self).__init__(**kwargs)
        if not isdefined(self.inputs.output_query):
            self.inputs.output_query = {'bold': {'datatype': 'func', 'suffix': 'bold', 'extension': ['nii', '.nii.gz']}, 'T1w': {'datatype': 'anat', 'suffix': 'T1w', 'extension': ['nii', '.nii.gz']}}
        if infields is None:
            from bids import layout as bidslayout
            bids_config = join(dirname(bidslayout.__file__), 'config', 'bids.json')
            bids_config = json.load(open(bids_config, 'r'))
            infields = [i['name'] for i in bids_config['entities']]
        self._infields = infields or []
        undefined_traits = {}
        for key in self._infields:
            self.inputs.add_trait(key, traits.Any)
            undefined_traits[key] = kwargs[key] if key in kwargs else Undefined
        self.inputs.trait_set(trait_change_notify=False, **undefined_traits)

    def _list_outputs(self):
        from bids import BIDSLayout
        if isdefined(self.inputs.load_layout):
            layout = BIDSLayout.load(self.inputs.load_layout)
        else:
            layout = BIDSLayout(self.inputs.base_dir, derivatives=self.inputs.index_derivatives)
        if isdefined(self.inputs.extra_derivatives):
            layout.add_derivatives(self.inputs.extra_derivatives)
        filters = {}
        for key in self._infields:
            value = getattr(self.inputs, key)
            if isdefined(value):
                filters[key] = value
        outputs = {}
        for key, query in self.inputs.output_query.items():
            args = query.copy()
            args.update(filters)
            filelist = layout.get(return_type='file', **args)
            if len(filelist) == 0:
                msg = 'Output key: %s returned no files' % key
                if self.inputs.raise_on_empty:
                    raise IOError(msg)
                else:
                    iflogger.warning(msg)
                    filelist = Undefined
            outputs[key] = filelist
        return outputs

    def _add_output_traits(self, base):
        return add_traits(base, list(self.inputs.output_query.keys()))