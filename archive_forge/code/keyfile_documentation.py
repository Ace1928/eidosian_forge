import os
from boto.exception import StorageResponseError

Wrapper class to expose a Key being read via a partial implementaiton of the
Python file interface. The only functions supported are those needed for seeking
in a Key open for reading.
