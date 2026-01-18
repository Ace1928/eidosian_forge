from __future__ import annotations
import json
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.core.spectrum import Spectrum
from pymatgen.core.structure import Structure
from pymatgen.util.plotting import add_fig_kwargs
from pymatgen.vis.plotters import SpectrumPlotter
Return an instance of the Spectrum plotter containing the different requested components.

        Arguments:
            components: A list with the components of the dielectric tensor to plot.
                        Can be either two indexes or a string like 'xx' to plot the (0,0) component
            reim: If 're' (im) is present in the string plots the real (imaginary) part of the dielectric tensor
            broad (float): a list of broadenings or a single broadening for the phonon peaks. Defaults to 0.00005.
            emin (float): minimum energy in which to obtain the spectra. Defaults to 0.
            emax (float): maximum energy in which to obtain the spectra. Defaults to None.
            divs: number of frequency samples between emin and emax
            **kwargs: Passed to IRDielectricTensor.get_spectrum()
        