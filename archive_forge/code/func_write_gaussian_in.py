import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def write_gaussian_in(fd, atoms, properties=['energy'], method=None, basis=None, fitting_basis=None, output_type='P', basisfile=None, basis_set=None, xc=None, charge=None, mult=None, extra=None, ioplist=None, addsec=None, spinlist=None, zefflist=None, qmomlist=None, nmagmlist=None, znuclist=None, radnuclearlist=None, **params):
    """
    Generates a Gaussian input file

    Parameters
    -----------
    fd: file-like
        where the Gaussian input file will be written
    atoms: Atoms
        Structure to write to the input file
    properties: list
        Properties to calculate
    method: str
        Level of theory to use, e.g. ``hf``, ``ccsd``, ``mp2``, or ``b3lyp``.
        Overrides ``xc`` (see below).
    xc: str
        Level of theory to use. Translates several XC functionals from
        their common name (e.g. ``PBE``) to their internal Gaussian name
        (e.g. ``PBEPBE``).
    basis: str
        The basis set to use. If not provided, no basis set will be requested,
        which usually results in ``STO-3G``. Maybe omitted if basisfile is set
        (see below).
    fitting_basis: str
        The name of the fitting basis set to use.
    output_type: str
        Level of output to record in the Gaussian
        output file - this may be ``N``- normal or ``P`` -
        additional.
    basisfile: str
        The name of the basis file to use. If a value is provided, basis may
        be omitted (it will be automatically set to 'gen')
    basis_set: str
        The basis set definition to use. This is an alternative
        to basisfile, and would be the same as the contents
        of such a file.
    charge: int
        The system charge. If not provided, it will be automatically
        determined from the ``Atoms`` object’s initial_charges.
    mult: int
        The system multiplicity (``spin + 1``). If not provided, it will be
        automatically determined from the ``Atoms`` object’s
        ``initial_magnetic_moments``.
    extra: str
        Extra lines to be included in the route section verbatim.
        It should not be necessary to use this, but it is included for
        backwards compatibility.
    ioplist: list
        A collection of IOPs definitions to be included in the route line.
    addsec: str
        Text to be added after the molecular geometry specification, e.g. for
        defining masses with ``freq=ReadIso``.
    spinlist: list
        A list of nuclear spins to be added into the nuclear
        propeties section of the molecule specification.
    zefflist: list
        A list of effective charges to be added into the nuclear
        propeties section of the molecule specification.
    qmomlist: list
        A list of nuclear quadropole moments to be added into
        the nuclear propeties section of the molecule
        specification.
    nmagmlist: list
        A list of nuclear magnetic moments to be added into
        the nuclear propeties section of the molecule
        specification.
    znuclist: list
        A list of nuclear charges to be added into the nuclear
        propeties section of the molecule specification.
    radnuclearlist: list
        A list of nuclear radii to be added into the nuclear
        propeties section of the molecule specification.
    params: dict
        Contains any extra keywords and values that will be included in either
        the link0 section or route section of the gaussian input file.
        To be included in the link0 section, the keyword must be one of the
        following: ``mem``, ``chk``, ``oldchk``, ``schk``, ``rwf``,
        ``oldmatrix``, ``oldrawmatrix``, ``int``, ``d2e``, ``save``,
        ``nosave``, ``errorsave``, ``cpu``, ``nprocshared``, ``gpucpu``,
        ``lindaworkers``, ``usessh``, ``ssh``, ``debuglinda``.
        Any other keywords will be placed (along with their values) in the
        route section.
    """
    params = deepcopy(params)
    if properties is None:
        properties = ['energy']
    output_type = _format_output_type(output_type)
    if basis is None:
        if basisfile is not None or basis_set is not None:
            basis = 'gen'
    if method is None:
        if xc is not None:
            method = _xc_to_method.get(xc.lower(), xc)
    if method is not None:
        _check_problem_methods(method)
    if charge is None:
        charge = atoms.get_initial_charges().sum()
    if mult is None:
        mult = atoms.get_initial_magnetic_moments().sum() + 1
    out = []
    params, link0_list = _pop_link0_params(params)
    out.extend(link0_list)
    out.append(_format_method_basis(output_type, method, basis, fitting_basis))
    params.pop('isolist', None)
    out.extend(_format_route_params(params))
    if ioplist is not None:
        out.append('IOP(' + ', '.join(ioplist) + ')')
    if extra is not None:
        out.append(extra)
    if 'forces' in properties and 'force' not in params:
        out.append('force')
    out += ['', 'Gaussian input prepared by ASE', '', '{:.0f} {:.0f}'.format(charge, mult)]
    nuclear_props = {'spin': spinlist, 'zeff': zefflist, 'qmom': qmomlist, 'nmagm': nmagmlist, 'znuc': znuclist, 'radnuclear': radnuclearlist}
    nuclear_props = {k: v for k, v in nuclear_props.items() if v is not None}
    molecule_spec = _get_molecule_spec(atoms, nuclear_props)
    for line in molecule_spec:
        out.append(line)
    out.extend(_format_basis_set(basis, basisfile, basis_set))
    out.extend(_format_addsec(addsec))
    out += ['', '']
    fd.write('\n'.join(out))