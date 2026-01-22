import enum
import typing
class CredentialsExpiredError(SpnegoError):
    ERROR_CODE = ErrorCode.credentials_expired
    _BASE_MESSAGE = 'The referenced credentials have expired'
    _GSSAPI_CODE = 720896
    _SSPI_CODE = -1073741711