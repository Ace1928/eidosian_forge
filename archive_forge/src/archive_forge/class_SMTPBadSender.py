from typing import Optional
class SMTPBadSender(SMTPAddressError):

    def __init__(self, addr, code=550, resp='Sender not acceptable'):
        SMTPAddressError.__init__(self, addr, code, resp)