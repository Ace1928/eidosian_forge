from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Spin
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
class BztPlotter:
    """Plotter to plot transport properties, interpolated bands along some high
    symmetry k-path, and DOS.

    Example:
        bztPlotter = BztPlotter(bztTransp,bztInterp)
        fig = self.bztPlotter.plot_props('S', 'mu', 'temp', temps=[300, 500])
        fig.show()
    """

    def __init__(self, bzt_transP=None, bzt_interp=None) -> None:
        """Placeholder.

        TODO: missing docstrings for __init__

        Args:
            bzt_transP (_type_, optional): _description_. Defaults to None.
            bzt_interp (_type_, optional): _description_. Defaults to None.
        """
        self.bzt_transP = bzt_transP
        self.bzt_interp = bzt_interp

    def plot_props(self, prop_y, prop_x, prop_z='temp', output='avg_eigs', dop_type='n', doping=None, temps=None, xlim=(-2, 2), ax: plt.Axes=None):
        """Function to plot the transport properties.

        Args:
            prop_y: property to plot among ("Conductivity","Seebeck","Kappa","Carrier_conc",
                "Hall_carrier_conc_trace"). Abbreviations are possible, like "S" for "Seebeck"
            prop_x: independent variable in the x-axis among ('mu','doping','temp')
            prop_z: third variable to plot multiple curves ('doping','temp')
            output: 'avg_eigs' to plot the average of the eigenvalues of the properties
                tensors; 'eigs' to plot the three eigenvalues of the properties
                tensors.
            dop_type: 'n' or 'p' to specify the doping type in plots that use doping
                levels as prop_x or prop_z
            doping: list of doping level to plot, useful to reduce the number of curves
                when prop_z='doping'
            temps: list of temperatures to plot, useful to reduce the number of curves
                when prop_z='temp'
            xlim: chemical potential range in eV, useful when prop_x='mu'
            ax: figure.axes where to plot. If None, a new figure is produced.

        Returns:
            plt.Axes: matplotlib Axes object

        Example:
            bztPlotter.plot_props('S','mu','temp',temps=[600,900,1200]).show()
            more example are provided in the notebook
            "How to use Boltztra2 interface.ipynb".
        """
        props = ('Conductivity', 'Seebeck', 'Kappa', 'Effective_mass', 'Power_Factor', 'Carrier_conc', 'Hall_carrier_conc_trace')
        props_lbl = ('Conductivity', 'Seebeck', '$K_{el}$', 'Effective mass', 'Power Factor', 'Carrier concentration', 'Hall carrier conc.')
        props_unit = ('$(\\mathrm{S\\,m^{-1}})$', '($\\mu$V/K)', '$(W / (m \\cdot K))$', '$(m_e)$', '$( mW / (m\\cdot K^2)$', '$(cm^{-3})$', '$(cm^{-3})$')
        props_short = [p[:len(prop_y)] for p in props]
        if prop_y not in props_short:
            raise BoltztrapError('prop_y not valid')
        if prop_x not in ('mu', 'doping', 'temp'):
            raise BoltztrapError('prop_x not valid')
        if prop_z not in ('doping', 'temp'):
            raise BoltztrapError('prop_z not valid')
        idx_prop = props_short.index(prop_y)
        leg_title = ''
        mu = self.bzt_transP.mu_r_eV
        if prop_z == 'doping' and prop_x == 'temp':
            p_array = getattr(self.bzt_transP, f'{props[idx_prop]}_doping')
        else:
            p_array = getattr(self.bzt_transP, f'{props[idx_prop]}_{prop_x}')
        if ax is None:
            plt.figure(figsize=(10, 8))
        temps_all = self.bzt_transP.temp_r.tolist()
        if temps is None:
            temps = self.bzt_transP.temp_r.tolist()
        if isinstance(self.bzt_transP.doping, np.ndarray):
            doping_all = self.bzt_transP.doping.tolist()
            if doping is None:
                doping = doping_all
        if idx_prop in [5, 6]:
            if prop_z == 'temp' and prop_x == 'mu':
                for temp in temps:
                    ti = temps_all.index(temp)
                    prop_out = p_array[ti] if idx_prop == 6 else np.abs(p_array[ti])
                    plt.semilogy(mu, prop_out, label=f'{temp} K')
                plt.xlabel('$\\mu$ (eV)', fontsize=30)
                plt.xlim(xlim)
            else:
                raise BoltztrapError('only prop_x=mu and prop_z=temp are                     available for c.c. and Hall c.c.!')
        elif prop_z == 'temp' and prop_x == 'mu':
            for temp in temps:
                ti = temps_all.index(temp)
                prop_out = np.linalg.eigh(p_array[ti])[0]
                if output == 'avg_eigs':
                    plt.plot(mu, prop_out.mean(axis=1), label=f'{temp} K')
                elif output == 'eigs':
                    for i in range(3):
                        plt.plot(mu, prop_out[:, i], label=f'eig {i} {temp} K')
            plt.xlabel('$\\mu$ (eV)', fontsize=30)
            plt.xlim(xlim)
        elif prop_z == 'temp' and prop_x == 'doping':
            for temp in temps:
                ti = temps_all.index(temp)
                prop_out = np.linalg.eigh(p_array[dop_type][ti])[0]
                if output == 'avg_eigs':
                    plt.semilogx(doping_all, prop_out.mean(axis=1), 's-', label=f'{temp} K')
                elif output == 'eigs':
                    for i in range(3):
                        plt.plot(doping_all, prop_out[:, i], 's-', label=f'eig {i} {temp} K')
            plt.xlabel('Carrier conc. $cm^{-3}$', fontsize=30)
            leg_title = dop_type + '-type'
        elif prop_z == 'doping' and prop_x == 'temp':
            for dop in doping:
                di = doping_all.index(dop)
                prop_out = np.linalg.eigh(p_array[dop_type][:, di])[0]
                if output == 'avg_eigs':
                    plt.plot(temps_all, prop_out.mean(axis=1), 's-', label=f'{dop} $cm^{-3}$')
                elif output == 'eigs':
                    for i in range(3):
                        plt.plot(temps_all, prop_out[:, i], 's-', label=f'eig {i} {dop} $cm^{{-3}}$')
            plt.xlabel('Temperature (K)', fontsize=30)
            leg_title = f'{dop_type}-type'
        plt.ylabel(f'{props_lbl[idx_prop]} {props_unit[idx_prop]}', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(title=leg_title if leg_title != '' else '', fontsize=15)
        plt.tight_layout()
        plt.grid()
        return ax

    def plot_bands(self):
        """Plot a band structure on symmetry line using BSPlotter()."""
        if self.bzt_interp is None:
            raise BoltztrapError('BztInterpolator not present')
        sbs = self.bzt_interp.get_band_structure()
        return BSPlotter(sbs).get_plot()

    def plot_dos(self, T=None, npoints=10000):
        """Plot the total Dos using DosPlotter()."""
        if self.bzt_interp is None:
            raise BoltztrapError('BztInterpolator not present')
        tdos = self.bzt_interp.get_dos(T=T, npts_mu=npoints)
        dosPlotter = DosPlotter()
        dosPlotter.add_dos('Total', tdos)
        return dosPlotter