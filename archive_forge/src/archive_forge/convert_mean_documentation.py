import argparse
import numpy as np
import mxnet as mx
import caffe_parser
Convert caffe mean

    Parameters
    ----------
    binaryproto_fname : str
        Filename of the mean
    output : str, optional
        Save the mean into mxnet's format

    Returns
    -------
    NDArray
        Mean in ndarray
    