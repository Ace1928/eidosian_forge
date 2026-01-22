import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
Calculate isolation by distance statistics for haploid data.

        See _calc_ibd for parameter details.

        Note that each pop can only have a single individual and
        the individual name has to be the sample coordinates.
        