from suds import *
from suds.client import Client

        Get an instance of an enumeration defined in the WSDL by name.
        @param name: The name of a enumeration defined in the WSDL.
        @type name: str
        @return: An instance on success, else None
        @rtype: L{sudsobject.Object}
        