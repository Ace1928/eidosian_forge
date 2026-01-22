from typing import Optional
class SMTPBadRcpt(SMTPAddressError):

    def __init__(self, addr, code=550, resp='Cannot receive for specified address'):
        SMTPAddressError.__init__(self, addr, code, resp)