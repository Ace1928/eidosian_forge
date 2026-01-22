from . import xml8
class Serializer_v6(xml8.Serializer_v8):
    """This serialiser supports rich roots.

    While its inventory format number is 6, its revision format is 5.
    Its inventory_sha1 may be inaccurate-- the inventory may have been
    converted from format 5 or 7 without updating the sha1.
    """
    format_num = b'6'
    revision_format_num = b'5'