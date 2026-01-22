class StdinNotImplementedError(IPythonCoreError, NotImplementedError):
    """raw_input was requested in a context where it is not supported

    For use in IPython kernels, where only some frontends may support
    stdin requests.
    """