from .import_onnx import GraphProto

    Imports the ONNX model files, passed as a parameter, into Gluon SymbolBlock object.

    Parameters
    ----------
    model_file : str
        ONNX model file name
    ctx : Context or list of Context
        Loads the model into one or many context(s).

    Returns
    -------
    sym_block : :class:`~mxnet.gluon.SymbolBlock`
        A SymbolBlock object representing the given model file.

    Notes
    -----
    This method is available when you ``import mxnet.contrib.onnx``

    