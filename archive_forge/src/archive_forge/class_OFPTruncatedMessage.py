class OFPTruncatedMessage(OSKenException):
    message = 'truncated message: %(orig_ex)s'

    def __init__(self, ofpmsg, residue, original_exception, msg=None, **kwargs):
        self.ofpmsg = ofpmsg
        self.residue = residue
        self.original_exception = original_exception
        kwargs['orig_ex'] = str(original_exception)
        super(OFPTruncatedMessage, self).__init__(msg, **kwargs)