from os_ken import exception
class UnexpectedMultiReply(_ExceptionBase):
    """Two or more replies are received for reply_muiti=False request."""
    message = 'Unexpected Multi replies %(result)s'