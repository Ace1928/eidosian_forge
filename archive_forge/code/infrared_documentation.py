from math import sqrt
from sys import stdout
import numpy as np
import ase.units as units
from ase.parallel import parprint, paropen
from ase.vibrations import Vibrations
Write out infrared spectrum to file.

        First column is the wavenumber in cm^-1, the second column the
        absolute infrared intensities, and
        the third column the absorbance scaled so that data runs
        from 1 to 0. Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1.