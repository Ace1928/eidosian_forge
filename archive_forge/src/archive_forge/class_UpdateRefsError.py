import binascii
class UpdateRefsError(GitProtocolError):
    """The server reported errors updating refs."""

    def __init__(self, *args, **kwargs):
        self.ref_status = kwargs.pop('ref_status')
        super(UpdateRefsError, self).__init__(*args, **kwargs)