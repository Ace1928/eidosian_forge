class ExecutableMissing(BzrError):
    _fmt = '%(exe_name)s could not be found on this machine'

    def __init__(self, exe_name):
        BzrError.__init__(self, exe_name=exe_name)