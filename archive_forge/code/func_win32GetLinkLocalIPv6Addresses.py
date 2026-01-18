from ctypes import (  # type: ignore[attr-defined]
from socket import AF_INET6, SOCK_STREAM, socket
def win32GetLinkLocalIPv6Addresses():
    """
    Return a list of strings in colon-hex format representing all the link local
    IPv6 addresses available on the system, as reported by
    I{WSAIoctl}/C{SIO_ADDRESS_LIST_QUERY}.
    """
    s = socket(AF_INET6, SOCK_STREAM)
    size = 4096
    retBytes = c_int()
    for i in range(2):
        buf = create_string_buffer(size)
        ret = WSAIoctl(s.fileno(), SIO_ADDRESS_LIST_QUERY, 0, 0, buf, size, byref(retBytes), 0, 0)
        if ret and retBytes.value:
            size = retBytes.value
        else:
            break
    if ret:
        raise RuntimeError('WSAIoctl failure')
    addrList = cast(buf, POINTER(make_SAL(0)))
    addrCount = addrList[0].iAddressCount
    addrList = cast(buf, POINTER(make_SAL(addrCount)))
    addressStringBufLength = 1024
    addressStringBuf = create_unicode_buffer(addressStringBufLength)
    retList = []
    for i in range(addrList[0].iAddressCount):
        retBytes.value = addressStringBufLength
        address = addrList[0].Address[i]
        ret = WSAAddressToString(address.lpSockaddr, address.iSockaddrLength, 0, addressStringBuf, byref(retBytes))
        if ret:
            raise RuntimeError('WSAAddressToString failure')
        retList.append(wstring_at(addressStringBuf))
    return [addr for addr in retList if '%' in addr]