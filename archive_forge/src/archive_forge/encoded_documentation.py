from suds import *
from suds.mx import *
from suds.mx.literal import Literal
from suds.mx.typer import Typer
from suds.sudsobject import Factory, Object
from suds.xsd.query import TypeQuery

        Cast the I{untyped} list items found in content I{value}.
        Each items contained in the list is checked for XSD type information.
        Items (values) that are I{untyped}, are replaced with suds objects and
        type I{metadata} is added.
        @param content: The content holding the collection.
        @type content: L{Content}
        @return: self
        @rtype: L{Encoded}
        