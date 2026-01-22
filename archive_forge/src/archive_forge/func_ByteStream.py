import io
from fixtures import Fixture
def ByteStream(detail_name):
    """Provide a file-like object that accepts bytes and expose as a detail.

    :param detail_name: The name of the detail.
    :return: A fixture which has an attribute `stream` containing the file-like
        object.
    """
    return Stream(detail_name, _byte_stream_factory)