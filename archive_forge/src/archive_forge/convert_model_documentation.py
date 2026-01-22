from __future__ import print_function
import argparse
import sys
import re
import numpy as np
import caffe_parser
import mxnet as mx
from convert_symbol import convert_symbol
Convert caffe model

    Parameters
    ----------

    prototxt_fname : str
         Filename of the prototxt model definition
    caffemodel_fname : str
         Filename of the binary caffe model
    output_prefix : str, optinoal
         If given, then save the converted MXNet into output_prefx+'.json' and
         output_prefx+'.params'

    Returns
    -------
    sym : Symbol
         Symbol convereted from prototxt
    arg_params : list of NDArray
         Argument parameters
    aux_params : list of NDArray
         Aux parameters
    input_dim : tuple
         Input dimension
    