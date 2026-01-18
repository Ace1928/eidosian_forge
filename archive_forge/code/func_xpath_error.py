import locale
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from .namespaces import XQT_ERRORS_NAMESPACE
from .datatypes import QName
def xpath_error(code: Union[str, QName], message_or_error: Union[None, str, Exception]=None, token: Optional['Token[Any]']=None, namespaces: Optional[Dict[str, str]]=None) -> ElementPathError:
    """
    Returns an XPath error instance related with a code. An XPath/XQuery/XSLT error code
    (ref: http://www.w3.org/2005/xqt-errors) is an alphanumeric token starting with four
    uppercase letters and ending with four digits.

    :param code: the error code.
    :param message_or_error: an optional custom message or related exception.
    :param token: an optional token instance.
    :param namespaces: an optional namespace mapping for finding the prefix     related with the namespace 'http://www.w3.org/2005/xqt-errors'.
    For default the prefix 'err' is used.
    """
    if isinstance(code, QName):
        namespace = code.uri
        if namespace:
            pcode, code = (code.qname, code.local_name)
        else:
            pcode, code = (code.braced_uri_name, code.local_name)
    else:
        namespace = XQT_ERRORS_NAMESPACE
        if not namespaces or namespaces.get('err') == XQT_ERRORS_NAMESPACE:
            prefix = 'err'
        else:
            for prefix, uri in namespaces.items():
                if uri == XQT_ERRORS_NAMESPACE:
                    break
            else:
                prefix = 'err'
        if code.startswith('{'):
            try:
                namespace, code = code[1:].split('}')
            except ValueError:
                message = '{!r} is not an xs:QName'.format(code)
                raise ElementPathValueError(message, 'err:XPTY0004', token)
            else:
                pcode = f'{prefix}:{code}' if prefix else code
        elif ':' not in code:
            pcode = f'{prefix}:{code}' if prefix else code
        elif code.startswith(f'{prefix}:') and code.count(':') == 1:
            pcode, code = (code, code.split(':')[1])
        else:
            message = '%r is not an XPath error code' % code
            raise ElementPathValueError(message, 'err:XPTY0004', token)
        if namespace != XQT_ERRORS_NAMESPACE:
            message = 'invalid namespace {!r}'.format(namespace)
            raise ElementPathValueError(message, 'err:XPTY0004', token)
    try:
        error_class, default_message = XPATH_ERROR_CODES[code]
    except KeyError:
        if namespace == XQT_ERRORS_NAMESPACE:
            message = f'unknown XPath error code {code}'
            raise ElementPathValueError(message, 'err:XPTY0004', token) from None
        else:
            error_class = ElementPathError
            default_message = 'custom XPath error'
    if message_or_error is None:
        message = default_message
    elif isinstance(message_or_error, str):
        message = message_or_error
    elif isinstance(message_or_error, ElementPathError):
        message = message_or_error.message
    else:
        message = str(message_or_error)
    return error_class(message, pcode, token)