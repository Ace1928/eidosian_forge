from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional
def var_to_xml(val, name, trim_if_too_big=True, additional_in_xml='', evaluate_full_value=True):
    """ single variable or dictionary to xml representation """
    type_name, type_qualifier, is_exception_on_eval, resolver, value = get_variable_details(val, evaluate_full_value)
    scope = get_var_scope(name, val, '', True)
    try:
        name = quote(name, '/>_= ')
    except:
        pass
    xml = '<var name="%s" type="%s" ' % (make_valid_xml_value(name), make_valid_xml_value(type_name))
    if type_qualifier:
        xml_qualifier = 'qualifier="%s"' % make_valid_xml_value(type_qualifier)
    else:
        xml_qualifier = ''
    if value:
        if len(value) > MAXIMUM_VARIABLE_REPRESENTATION_SIZE and trim_if_too_big:
            value = value[0:MAXIMUM_VARIABLE_REPRESENTATION_SIZE]
            value += '...'
        xml_value = ' value="%s"' % make_valid_xml_value(quote(value, '/>_= '))
    else:
        xml_value = ''
    if is_exception_on_eval:
        xml_container = ' isErrorOnEval="True"'
    elif resolver is not None:
        xml_container = ' isContainer="True"'
    else:
        xml_container = ''
    if scope:
        return ''.join((xml, xml_qualifier, xml_value, xml_container, additional_in_xml, ' scope="', scope, '"', ' />\n'))
    else:
        return ''.join((xml, xml_qualifier, xml_value, xml_container, additional_in_xml, ' />\n'))