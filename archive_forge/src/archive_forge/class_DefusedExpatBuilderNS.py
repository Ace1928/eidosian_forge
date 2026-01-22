from __future__ import print_function, absolute_import
from xml.dom.expatbuilder import ExpatBuilder as _ExpatBuilder
from xml.dom.expatbuilder import Namespaces as _Namespaces
from .common import DTDForbidden, EntitiesForbidden, ExternalReferenceForbidden
class DefusedExpatBuilderNS(_Namespaces, DefusedExpatBuilder):
    """Defused document builder that supports namespaces."""

    def install(self, parser):
        DefusedExpatBuilder.install(self, parser)
        if self._options.namespace_declarations:
            parser.StartNamespaceDeclHandler = self.start_namespace_decl_handler

    def reset(self):
        DefusedExpatBuilder.reset(self)
        self._initNamespaces()