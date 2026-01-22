from __future__ import annotations
import re
import subprocess
import warnings
from shutil import which
import requests
from monty.dev import requires
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
Queries the COD for structures by formula. Requires mysql executable to
        be in the path.

        Args:
            formula (str): Chemical formula.
            kwargs: All kwargs supported by Structure.from_str.

        Returns:
            A list of dict of the format [{"structure": Structure, "cod_id": int, "sg": "P n m a"}]
        