from __future__ import print_function, absolute_import
import sys
import warnings
from xml.etree.ElementTree import ParseError
from xml.etree.ElementTree import TreeBuilder as _TreeBuilder
from xml.etree.ElementTree import parse as _parse
from xml.etree.ElementTree import tostring
from .common import PY3
from .common import (
class DefusedXMLParser(_XMLParser):

    def __init__(self, html=_sentinel, target=None, encoding=None, forbid_dtd=False, forbid_entities=True, forbid_external=True):
        _XMLParser.__init__(self, target=target, encoding=encoding)
        if html is not _sentinel:
            if html:
                raise TypeError("'html=True' is no longer supported.")
            else:
                warnings.warn("'html' keyword argument is no longer supported. Pass in arguments as keyword arguments.", category=DeprecationWarning)
        self.forbid_dtd = forbid_dtd
        self.forbid_entities = forbid_entities
        self.forbid_external = forbid_external
        if PY3:
            parser = self.parser
        else:
            parser = self._parser
        if self.forbid_dtd:
            parser.StartDoctypeDeclHandler = self.defused_start_doctype_decl
        if self.forbid_entities:
            parser.EntityDeclHandler = self.defused_entity_decl
            parser.UnparsedEntityDeclHandler = self.defused_unparsed_entity_decl
        if self.forbid_external:
            parser.ExternalEntityRefHandler = self.defused_external_entity_ref_handler

    def defused_start_doctype_decl(self, name, sysid, pubid, has_internal_subset):
        raise DTDForbidden(name, sysid, pubid)

    def defused_entity_decl(self, name, is_parameter_entity, value, base, sysid, pubid, notation_name):
        raise EntitiesForbidden(name, value, base, sysid, pubid, notation_name)

    def defused_unparsed_entity_decl(self, name, base, sysid, pubid, notation_name):
        raise EntitiesForbidden(name, None, base, sysid, pubid, notation_name)

    def defused_external_entity_ref_handler(self, context, base, sysid, pubid):
        raise ExternalReferenceForbidden(context, base, sysid, pubid)