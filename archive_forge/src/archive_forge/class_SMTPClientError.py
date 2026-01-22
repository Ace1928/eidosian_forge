from typing import Optional
class SMTPClientError(SMTPError):
    """
    Base class for SMTP client errors.
    """

    def __init__(self, code: int, resp: bytes, log: Optional[bytes]=None, addresses: Optional[object]=None, isFatal: bool=False, retry: bool=False):
        """
        @param code: The SMTP response code associated with this error.
        @param resp: The string response associated with this error.
        @param log: A string log of the exchange leading up to and including
            the error.
        @param isFatal: A boolean indicating whether this connection can
            proceed or not. If True, the connection will be dropped.
        @param retry: A boolean indicating whether the delivery should be
            retried. If True and the factory indicates further retries are
            desirable, they will be attempted, otherwise the delivery will be
            failed.
        """
        if isinstance(resp, str):
            resp = resp.encode('utf-8')
        if isinstance(log, str):
            log = log.encode('utf-8')
        self.code = code
        self.resp = resp
        self.log = log
        self.addresses = addresses
        self.isFatal = isFatal
        self.retry = retry

    def __str__(self) -> str:
        return self.__bytes__().decode('utf-8')

    def __bytes__(self) -> bytes:
        if self.code > 0:
            res = [f'{self.code:03d} '.encode() + self.resp]
        else:
            res = [self.resp]
        if self.log:
            res.append(self.log)
            res.append(b'')
        return b'\n'.join(res)