from __future__ import annotations
import functools
import itertools
import json
import logging
import math
import os
from collections import defaultdict
from operator import mul
from typing import TYPE_CHECKING
from monty.design_patterns import cached_class
from pymatgen.core import Species, get_el_sp
from pymatgen.util.due import Doi, due

        Returns charged balanced substitutions from a starting or ending
        composition.

        Args:
            composition:
                starting or ending composition
            to_this_composition:
                If true, substitutions with this as a final composition
                will be found. If false, substitutions with this as a
                starting composition will be found (these are slightly
                different)

        Returns:
            List of predictions in the form of dictionaries.
            If to_this_composition is true, the values of the dictionary
            will be from the list species. If false, the keys will be
            from that list.
        