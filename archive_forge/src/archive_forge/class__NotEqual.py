from io import BytesIO
from xml.dom import minidom as dom
from twisted.internet.protocol import FileWrapper
class _NotEqual:
    """
    A class the instances of which are equal to nothing.
    """

    def __eq__(self, other: object) -> bool:
        return False