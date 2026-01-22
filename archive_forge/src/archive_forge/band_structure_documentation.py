from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from ase.cli.main import CLIError
Plot band-structure.

    Read eigenvalues and k-points from file and plot result from
    band-structure calculation or interpolate
    from Monkhorst-Pack sampling to a given path (--path=PATH).

    Example:

        $ ase band-structure bandstructure.json -r -10 10
    