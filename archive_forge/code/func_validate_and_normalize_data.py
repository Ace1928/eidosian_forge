from __future__ import absolute_import, division, print_function
import json
import sys
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
def validate_and_normalize_data(data, fmt=None):
    """
    This function validates the data for given format (fmt).
    If the fmt is None it tires to guess the data format.
    Currently support data format checks are
    1) xml
    2) json
    3) xpath
    :param data: The data which should be validated and normalised.
    :param fmt: This is an optional argument which indicated the format
    of the data. Valid values are "xml", "json" and "xpath". If the value
    is None the format of the data will be guessed and returned in the output.
    :return:
        *  If the format identified is XML it parses the xml data and returns
           a tuple of lxml.etree.Element class object and the data format type
           which is "xml" in this case.

        *  If the format identified is JSON it parses the json data and returns
           a tuple of dict object and the data format type
           which is "json" in this case.

        *  If the format identified is XPATH it parses the XPATH data and returns
           a tuple of etree.XPath class object and the data format type
           which is "xpath" in this case. For this type lxml library is required
           to be installed.
    """
    if data is None:
        return (None, None)
    if isinstance(data, string_types):
        data = data.strip()
        if data.startswith('<') and data.endswith('>') or fmt == 'xml':
            try:
                result = fromstring(data)
                if fmt and fmt != 'xml':
                    raise Exception("Invalid format '%s'. Expected format is 'xml' for data '%s'" % (fmt, data))
                return (result, 'xml')
            except XMLSyntaxError as exc:
                if fmt == 'xml':
                    raise Exception("'%s' XML validation failed with error '%s'" % (data, to_native(exc, errors='surrogate_then_replace')))
                pass
            except Exception as exc:
                error = "'%s' recognized as XML but was not valid." % data
                raise Exception(error + to_native(exc, errors='surrogate_then_replace'))
        elif data.startswith('{') and data.endswith('}') or fmt == 'json':
            try:
                result = json.loads(data)
                if fmt and fmt != 'json':
                    raise Exception("Invalid format '%s'. Expected format is 'json' for data '%s'" % (fmt, data))
                return (result, 'json')
            except (TypeError, getattr(json.decoder, 'JSONDecodeError', ValueError)) as exc:
                if fmt == 'json':
                    raise Exception("'%s' JSON validation failed with error '%s'" % (data, to_native(exc, errors='surrogate_then_replace')))
            except Exception as exc:
                error = "'%s' recognized as JSON but was not valid." % data
                raise Exception(error + to_native(exc, errors='surrogate_then_replace'))
        else:
            try:
                if not HAS_LXML:
                    raise Exception(missing_required_lib('lxml'))
                result = etree.XPath(data)
                if fmt and fmt != 'xpath':
                    raise Exception("Invalid format '%s'. Expected format is 'xpath' for data '%s'" % (fmt, data))
                return (result, 'xpath')
            except etree.XPathSyntaxError as exc:
                if fmt == 'xpath':
                    raise Exception("'%s' XPath validation failed with error '%s'" % (data, to_native(exc, errors='surrogate_then_replace')))
                pass
            except Exception as exc:
                error = "'%s' recognized as Xpath but was not valid." % data
                raise Exception(error + to_native(exc, errors='surrogate_then_replace'))
    elif isinstance(data, dict):
        if fmt and fmt != 'json':
            raise Exception("Invalid format '%s'. Expected format is 'json' for data '%s'" % (fmt, data))
        try:
            result = json.loads(json.dumps(data))
            return (result, 'json')
        except (TypeError, getattr(json.decoder, 'JSONDecodeError', ValueError)) as exc:
            raise Exception("'%s' JSON validation failed with error '%s'" % (data, to_native(exc, errors='surrogate_then_replace')))
        except Exception as exc:
            error = "'%s' recognized as JSON but was not valid." % data
            raise Exception(error + to_native(exc, errors='surrogate_then_replace'))
    return (data, None)