class ZMQExitAutocall(ExitAutocall):
    """Exit IPython. Autocallable, so it needn't be explicitly called.
    
    Parameters
    ----------
    keep_kernel : bool
      If True, leave the kernel alive. Otherwise, tell the kernel to exit too
      (default).
    """

    def __call__(self, keep_kernel=False):
        self._ip.keepkernel_on_exit = keep_kernel
        self._ip.ask_exit()