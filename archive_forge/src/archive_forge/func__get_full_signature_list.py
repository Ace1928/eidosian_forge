import ctypes
import enum
import os
import platform
import sys
import numpy as np
def _get_full_signature_list(self):
    """Gets list of SignatureDefs in the model.

    Example,
    ```
    signatures = interpreter._get_full_signature_list()
    print(signatures)

    # {
    #   'add': {'inputs': {'x': 1, 'y': 0}, 'outputs': {'output_0': 4}}
    # }

    Then using the names in the signature list you can get a callable from
    get_signature_runner().
    ```

    Returns:
      A list of SignatureDef details in a dictionary structure.
      It is keyed on the SignatureDef method name, and the value holds
      dictionary of inputs and outputs.
    """
    return self._interpreter.GetSignatureDefs()