from typing import Optional
class SMTPServerError(SMTPError):

    def __init__(self, code, resp):
        self.code = code
        self.resp = resp

    def __str__(self) -> str:
        return '%.3d %s' % (self.code, self.resp)