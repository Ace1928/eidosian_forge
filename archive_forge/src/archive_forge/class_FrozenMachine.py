class FrozenMachine(AutomatonException):
    """Exception raised when a frozen machine is modified."""

    def __init__(self):
        super(FrozenMachine, self).__init__("Frozen machine can't be modified")