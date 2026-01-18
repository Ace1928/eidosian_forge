import mxnet as mx
Check index in idx2name to get corresponding param_name
        Parameters
        ----------
        index : int or str
            An unique index to identify the weight.
        Returns
        -------
        name : str
            Name of the Module parameter
        