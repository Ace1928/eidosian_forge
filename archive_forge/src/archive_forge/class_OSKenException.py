class OSKenException(Exception):
    message = 'An unknown exception'

    def __init__(self, msg=None, **kwargs):
        self.kwargs = kwargs
        if msg is None:
            msg = self.message
        try:
            msg = msg % kwargs
        except Exception:
            msg = self.message
        super(OSKenException, self).__init__(msg)