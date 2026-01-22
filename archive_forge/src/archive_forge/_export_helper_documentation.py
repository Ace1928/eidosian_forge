import os
import logging
import mxnet as mx
Loads the MXNet model file and
    returns MXNet symbol and params (weights).

    Parameters
    ----------
    json_path : str
        Path to the json file
    params_path : str
        Path to the params file

    Returns
    -------
    sym : MXNet symbol
        Model symbol object

    params : params object
        Model weights including both arg and aux params.
    