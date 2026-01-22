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
class DataFinder(IOBase):
    """Search for paths that match a given regular expression. Allows a less
    proscriptive approach to gathering input files compared to DataGrabber.
    Will recursively search any subdirectories by default. This can be limited
    with the min/max depth options.
    Matched paths are available in the output 'out_paths'. Any named groups of
    captured text from the regular expression are also available as outputs of
    the same name.

    Examples
    --------
    >>> from nipype.interfaces.io import DataFinder
    >>> df = DataFinder()
    >>> df.inputs.root_paths = '.'
    >>> df.inputs.match_regex = r'.+/(?P<series_dir>.+(qT1|ep2d_fid_T1).+)/(?P<basename>.+)\\.nii.gz'
    >>> result = df.run() # doctest: +SKIP
    >>> result.outputs.out_paths  # doctest: +SKIP
    ['./027-ep2d_fid_T1_Gd4/acquisition.nii.gz',
     './018-ep2d_fid_T1_Gd2/acquisition.nii.gz',
     './016-ep2d_fid_T1_Gd1/acquisition.nii.gz',
     './013-ep2d_fid_T1_pre/acquisition.nii.gz']
    >>> result.outputs.series_dir  # doctest: +SKIP
    ['027-ep2d_fid_T1_Gd4',
     '018-ep2d_fid_T1_Gd2',
     '016-ep2d_fid_T1_Gd1',
     '013-ep2d_fid_T1_pre']
    >>> result.outputs.basename  # doctest: +SKIP
    ['acquisition',
     'acquisition'
     'acquisition',
     'acquisition']

    """
    input_spec = DataFinderInputSpec
    output_spec = DynamicTraitedSpec
    _always_run = True

    def _match_path(self, target_path):
        for ignore_re in self.ignore_regexes:
            if ignore_re.search(target_path):
                return
        match = self.match_regex.search(target_path)
        if match is not None:
            match_dict = match.groupdict()
            if self.result is None:
                self.result = {'out_paths': []}
                for key in list(match_dict.keys()):
                    self.result[key] = []
            self.result['out_paths'].append(target_path)
            for key, val in list(match_dict.items()):
                self.result[key].append(val)

    def _run_interface(self, runtime):
        if isinstance(self.inputs.root_paths, (str, bytes)):
            self.inputs.root_paths = [self.inputs.root_paths]
        self.match_regex = re.compile(self.inputs.match_regex)
        if self.inputs.max_depth is Undefined:
            max_depth = None
        else:
            max_depth = self.inputs.max_depth
        if self.inputs.min_depth is Undefined:
            min_depth = 0
        else:
            min_depth = self.inputs.min_depth
        if self.inputs.ignore_regexes is Undefined:
            self.ignore_regexes = []
        else:
            self.ignore_regexes = [re.compile(regex) for regex in self.inputs.ignore_regexes]
        self.result = None
        for root_path in self.inputs.root_paths:
            root_path = os.path.normpath(os.path.expandvars(os.path.expanduser(root_path)))
            if os.path.isfile(root_path):
                if min_depth == 0:
                    self._match_path(root_path)
                continue
            for curr_dir, sub_dirs, files in os.walk(root_path):
                curr_depth = curr_dir.count(os.sep) - root_path.count(os.sep)
                if max_depth is not None and curr_depth >= max_depth:
                    sub_dirs[:] = []
                    files = []
                if curr_depth >= min_depth:
                    self._match_path(curr_dir)
                if curr_depth >= min_depth - 1:
                    for infile in files:
                        full_path = os.path.join(curr_dir, infile)
                        self._match_path(full_path)
        if self.inputs.unpack_single and len(self.result['out_paths']) == 1:
            for key, vals in list(self.result.items()):
                self.result[key] = vals[0]
        else:
            for key in list(self.result.keys()):
                if key == 'out_paths':
                    continue
                sort_tuples = human_order_sorted(list(zip(self.result['out_paths'], self.result[key])))
                self.result[key] = [x for _, x in sort_tuples]
            self.result['out_paths'] = human_order_sorted(self.result['out_paths'])
        if not self.result:
            raise RuntimeError('Regular expression did not match any files!')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs.update(self.result)
        return outputs