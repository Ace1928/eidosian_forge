from suds import *
from suds.mx import *
from suds.mx.core import Core
from suds.mx.typer import Typer
from suds.resolver import Frame, GraphResolver
from suds.sax.element import Element
from suds.sudsobject import Factory
from logging import getLogger

        Attribute ordering defined in the specified XSD type information.

        @param type: XSD type object.
        @type type: L{SchemaObject}
        @return: An ordered list of attribute names.
        @rtype: list

        