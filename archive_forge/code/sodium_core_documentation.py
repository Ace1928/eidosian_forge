from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure

    Initializes sodium, picking the best implementations available for this
    machine.
    