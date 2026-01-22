class InterruptedError(Error):
    """Raised by APIProxy.Wait() when the wait is interrupted by an uncaught
  exception from some callback, not necessarily associated with the RPC in
  question."""

    def __init__(self, exception, rpc):
        self.args = ('The Wait() request was interrupted by an exception from another callback:', exception)
        self.__rpc = rpc
        self.__exception = exception

    @property
    def rpc(self):
        return self.__rpc

    @property
    def exception(self):
        return self.__exception