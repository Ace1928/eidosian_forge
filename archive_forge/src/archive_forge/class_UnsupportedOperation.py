class UnsupportedOperation(BzrError):
    _fmt = 'The method %(mname)s is not supported on objects of type %(tname)s.'

    def __init__(self, method, method_self):
        self.method = method
        self.mname = method.__name__
        self.tname = type(method_self).__name__