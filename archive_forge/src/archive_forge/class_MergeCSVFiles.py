import os
import os.path as op
import nibabel as nb
import numpy as np
from math import floor, ceil
import itertools
import warnings
from .. import logging
from . import metrics as nam
from ..interfaces.base import (
from ..utils.filemanip import fname_presuffix, split_filename, ensure_list
from . import confounds
class MergeCSVFiles(BaseInterface):
    """
    Merge several CSV files into a single CSV file.

    This interface is designed to facilitate data loading in the R environment.
    If provided, it will also incorporate column heading names into the
    resulting CSV file.
    CSV files are easily loaded in R, for use in statistical processing.
    For further information, see cran.r-project.org/doc/manuals/R-data.pdf

    Example
    -------
    >>> from nipype.algorithms import misc
    >>> mat2csv = misc.MergeCSVFiles()
    >>> mat2csv.inputs.in_files = ['degree.mat','clustering.mat']
    >>> mat2csv.inputs.column_headings = ['degree','clustering']
    >>> mat2csv.run() # doctest: +SKIP

    """
    input_spec = MergeCSVFilesInputSpec
    output_spec = MergeCSVFilesOutputSpec

    def _run_interface(self, runtime):
        extraheadingBool = False
        extraheading = ''
        rowheadingsBool = False
        '\n        This block defines the column headings.\n        '
        if isdefined(self.inputs.column_headings):
            iflogger.info('Column headings have been provided:')
            headings = self.inputs.column_headings
        else:
            iflogger.info('Column headings not provided! Pulled from input filenames:')
            headings = remove_identical_paths(self.inputs.in_files)
        if isdefined(self.inputs.extra_field):
            if isdefined(self.inputs.extra_column_heading):
                extraheading = self.inputs.extra_column_heading
                iflogger.info('Extra column heading provided: %s', extraheading)
            else:
                extraheading = 'type'
                iflogger.info('Extra column heading was not defined. Using "type"')
            headings.append(extraheading)
            extraheadingBool = True
        if len(self.inputs.in_files) == 1:
            iflogger.warning('Only one file input!')
        if isdefined(self.inputs.row_headings):
            iflogger.info('Row headings have been provided. Adding "labels"column header.')
            prefix = '"{p}","'.format(p=self.inputs.row_heading_title)
            csv_headings = prefix + '","'.join(itertools.chain(headings)) + '"\n'
            rowheadingsBool = True
        else:
            iflogger.info('Row headings have not been provided.')
            csv_headings = '"' + '","'.join(itertools.chain(headings)) + '"\n'
        iflogger.info('Final Headings:')
        iflogger.info(csv_headings)
        '\n        Next we merge the arrays and define the output text file\n        '
        output_array = merge_csvs(self.inputs.in_files)
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.csv':
            ext = '.csv'
        out_file = op.abspath(name + ext)
        with open(out_file, 'w') as file_handle:
            file_handle.write(csv_headings)
        shape = np.shape(output_array)
        typelist = maketypelist(rowheadingsBool, shape, extraheadingBool, extraheading)
        fmt, output = makefmtlist(output_array, typelist, rowheadingsBool, shape, extraheadingBool)
        if rowheadingsBool:
            row_heading_list = self.inputs.row_headings
            row_heading_list_with_quotes = []
            for row_heading in row_heading_list:
                row_heading_with_quotes = '"' + row_heading + '"'
                row_heading_list_with_quotes.append(row_heading_with_quotes)
            row_headings = np.array(row_heading_list_with_quotes, dtype='|S40')
            output['heading'] = row_headings
        if isdefined(self.inputs.extra_field):
            extrafieldlist = []
            if len(shape) > 1:
                mx = shape[0]
            else:
                mx = 1
            for idx in range(0, mx):
                extrafieldlist.append(self.inputs.extra_field)
            iflogger.info(len(extrafieldlist))
            output[extraheading] = extrafieldlist
        iflogger.info(output)
        iflogger.info(fmt)
        with open(out_file, 'a') as file_handle:
            np.savetxt(file_handle, output, fmt, delimiter=',')
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _, name, ext = split_filename(self.inputs.out_file)
        if not ext == '.csv':
            ext = '.csv'
        out_file = op.abspath(name + ext)
        outputs['csv_file'] = out_file
        return outputs