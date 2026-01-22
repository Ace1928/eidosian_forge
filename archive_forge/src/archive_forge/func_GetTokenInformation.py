from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetTokenInformation(hTokenHandle, TokenInformationClass):
    if TokenInformationClass <= 0 or TokenInformationClass > MaxTokenInfoClass:
        raise ValueError('Invalid value for TokenInformationClass (%i)' % TokenInformationClass)
    if TokenInformationClass == TokenUser:
        TokenInformation = TOKEN_USER()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.User.Sid.value
    if TokenInformationClass == TokenOwner:
        TokenInformation = TOKEN_OWNER()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.Owner.value
    if TokenInformationClass == TokenOwner:
        TokenInformation = TOKEN_PRIMARY_GROUP()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.PrimaryGroup.value
    if TokenInformationClass == TokenAppContainerSid:
        TokenInformation = TOKEN_APPCONTAINER_INFORMATION()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.TokenAppContainer.value
    if TokenInformationClass == TokenIntegrityLevel:
        TokenInformation = TOKEN_MANDATORY_LABEL()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return (TokenInformation.Label.Sid.value, TokenInformation.Label.Attributes)
    if TokenInformationClass == TokenOrigin:
        TokenInformation = TOKEN_ORIGIN()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.OriginatingLogonSession
    if TokenInformationClass == TokenType:
        TokenInformation = TOKEN_TYPE(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.value
    if TokenInformationClass == TokenElevation:
        TokenInformation = TOKEN_ELEVATION(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.value
    if TokenInformationClass == TokenElevation:
        TokenInformation = SECURITY_IMPERSONATION_LEVEL(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.value
    if TokenInformationClass in (TokenSessionId, TokenAppContainerNumber):
        TokenInformation = DWORD(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation.value
    if TokenInformationClass in (TokenSandBoxInert, TokenHasRestrictions, TokenUIAccess, TokenVirtualizationAllowed, TokenVirtualizationEnabled):
        TokenInformation = DWORD(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return bool(TokenInformation.value)
    if TokenInformationClass == TokenLinkedToken:
        TokenInformation = TOKEN_LINKED_TOKEN(0)
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenHandle(TokenInformation.LinkedToken.value, bOwnership=True)
    if TokenInformationClass == TokenStatistics:
        TokenInformation = TOKEN_STATISTICS()
        _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation)
        return TokenInformation
    raise NotImplementedError('TokenInformationClass(%i) not yet supported!' % TokenInformationClass)