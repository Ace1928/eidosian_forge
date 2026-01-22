import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def GetMolHash(all_layers, hash_scheme: HashScheme=HashScheme.ALL_LAYERS) -> str:
    """
    Generate a molecular hash using a specified set of layers.

    :param all_layers: a dictionary of layers
    :param hash_scheme: enum encoding information layers for the hash
    :return: hash for the given scheme constructed from the input layers
    """
    h = hashlib.sha1()
    for layer in hash_scheme.value:
        if layer in all_layers:
            h.update(all_layers[layer].encode())
    return h.hexdigest()