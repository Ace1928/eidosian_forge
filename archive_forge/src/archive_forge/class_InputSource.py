from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
class InputSource:
    """Encapsulation of the information needed by the XMLReader to
    read entities.

    This class may include information about the public identifier,
    system identifier, byte stream (possibly with character encoding
    information) and/or the character stream of an entity.

    Applications will create objects of this class for use in the
    XMLReader.parse method and for returning from
    EntityResolver.resolveEntity.

    An InputSource belongs to the application, the XMLReader is not
    allowed to modify InputSource objects passed to it from the
    application, although it may make copies and modify those."""

    def __init__(self, system_id=None):
        self.__system_id = system_id
        self.__public_id = None
        self.__encoding = None
        self.__bytefile = None
        self.__charfile = None

    def setPublicId(self, public_id):
        """Sets the public identifier of this InputSource."""
        self.__public_id = public_id

    def getPublicId(self):
        """Returns the public identifier of this InputSource."""
        return self.__public_id

    def setSystemId(self, system_id):
        """Sets the system identifier of this InputSource."""
        self.__system_id = system_id

    def getSystemId(self):
        """Returns the system identifier of this InputSource."""
        return self.__system_id

    def setEncoding(self, encoding):
        """Sets the character encoding of this InputSource.

        The encoding must be a string acceptable for an XML encoding
        declaration (see section 4.3.3 of the XML recommendation).

        The encoding attribute of the InputSource is ignored if the
        InputSource also contains a character stream."""
        self.__encoding = encoding

    def getEncoding(self):
        """Get the character encoding of this InputSource."""
        return self.__encoding

    def setByteStream(self, bytefile):
        """Set the byte stream (a Python file-like object which does
        not perform byte-to-character conversion) for this input
        source.

        The SAX parser will ignore this if there is also a character
        stream specified, but it will use a byte stream in preference
        to opening a URI connection itself.

        If the application knows the character encoding of the byte
        stream, it should set it with the setEncoding method."""
        self.__bytefile = bytefile

    def getByteStream(self):
        """Get the byte stream for this input source.

        The getEncoding method will return the character encoding for
        this byte stream, or None if unknown."""
        return self.__bytefile

    def setCharacterStream(self, charfile):
        """Set the character stream for this input source. (The stream
        must be a Python 2.0 Unicode-wrapped file-like that performs
        conversion to Unicode strings.)

        If there is a character stream specified, the SAX parser will
        ignore any byte stream and will not attempt to open a URI
        connection to the system identifier."""
        self.__charfile = charfile

    def getCharacterStream(self):
        """Get the character stream for this input source."""
        return self.__charfile