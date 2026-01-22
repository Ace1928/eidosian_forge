import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _FindFileContainingSymbolInDb(self, symbol):
    """Finds the file in descriptor DB containing the specified symbol.

    Args:
      symbol (str): The name of the symbol to search for.

    Returns:
      FileDescriptor: The file that contains the specified symbol.

    Raises:
      KeyError: if the file cannot be found in the descriptor database.
    """
    try:
        file_proto = self._internal_db.FindFileContainingSymbol(symbol)
    except KeyError as error:
        if self._descriptor_db:
            file_proto = self._descriptor_db.FindFileContainingSymbol(symbol)
        else:
            raise error
    if not file_proto:
        raise KeyError('Cannot find a file containing %s' % symbol)
    return self._ConvertFileProtoToFileDescriptor(file_proto)