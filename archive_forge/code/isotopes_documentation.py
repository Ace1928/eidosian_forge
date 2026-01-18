from urllib import request
Download isotope data from NIST public website.

    Relative atomic masses of individual isotopes their abundance
    (mole fraction) are compiled into a dictionary. Individual items can be
    indexed by the atomic number and mass number, e.g. titanium-48:

    >>> from ase.data.isotopes import download_isotope_data
    >>> isotopes = download_isotope_data()
    >>> isotopes[22][48]['mass']
    47.94794198
    >>> isotopes[22][48]['composition']
    0.7372
    