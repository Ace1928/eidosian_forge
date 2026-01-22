from __future__ import annotations
import abc
import collections
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, namedtuple
from typing import TYPE_CHECKING
from xml.etree import ElementTree as Et
import numpy as np
from monty.collections import AttrDict, Namespace
from monty.functools import lazy_property
from monty.itertools import iterator_from_slice
from monty.json import MontyDecoder, MSONable
from monty.os.path import find_exts
from tabulate import tabulate
from pymatgen.core import Element
from pymatgen.core.xcfunc import XcFunc
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class PawXmlSetup(Pseudo, PawPseudo):
    """Setup class for PawXml."""

    def __init__(self, filepath):
        """
        Args:
            filepath (str): Path to the XML file.
        """
        self.path = os.path.abspath(filepath)
        root = self.root
        self.paw_setup_version = root.get('version')
        atom_attrib = root.find('atom').attrib
        self._zatom = int(float(atom_attrib['Z']))
        self.core, self.valence = map(float, [atom_attrib['core'], atom_attrib['valence']])
        xc_info = root.find('xc_functional').attrib
        self.xc = XcFunc.from_type_name(xc_info['type'], xc_info['name'])
        pawr_element = root.find('PAW_radius')
        self._paw_radius = None
        if pawr_element is not None:
            self._paw_radius = float(pawr_element.attrib['rpaw'])
        self.valence_states = {}
        for node in root.find('valence_states'):
            attrib = AttrDict(node.attrib)
            assert attrib.id not in self.valence_states
            self.valence_states[attrib.id] = attrib
        self.rad_grids = {}
        for node in root.findall('radial_grid'):
            grid_params = node.attrib
            gid = grid_params['id']
            assert gid not in self.rad_grids
            self.rad_grids[gid] = self._eval_grid(grid_params)

    def __getstate__(self):
        """
        Return state is pickled as the contents for the instance.

        In this case we just remove the XML root element process since Element object cannot be pickled.
        """
        return {k: v for k, v in self.__dict__.items() if k != '_root'}

    @lazy_property
    def root(self):
        """Root tree of XML."""
        tree = Et.parse(self.filepath)
        return tree.getroot()

    @property
    def Z(self):
        return self._zatom

    @property
    def Z_val(self):
        """Number of valence electrons."""
        return self.valence

    @property
    def l_max(self):
        """Maximum angular momentum."""
        return

    @property
    def l_local(self):
        """Angular momentum used for the local part."""
        return

    @property
    def summary(self):
        """String summarizing the most important properties."""
        return ''

    @property
    def paw_radius(self):
        return self._paw_radius

    @property
    def supports_soc(self):
        """Here I assume that the ab-initio code can treat the SOC within the on-site approximation."""
        return True

    @staticmethod
    def _eval_grid(grid_params):
        """
        This function receives a dictionary with the parameters defining the
        radial mesh and returns a `ndarray` with the mesh.
        """
        eq = grid_params.get('eq').replace(' ', '')
        istart, iend = (int(grid_params.get('istart')), int(grid_params.get('iend')))
        indices = list(range(istart, iend + 1))
        if eq == 'r=a*exp(d*i)':
            a, d = (float(grid_params['a']), float(grid_params['d']))
            mesh = [a * np.exp(d * i) for i in indices]
        elif eq == 'r=a*i/(n-i)':
            a, n = (float(grid_params['a']), float(grid_params['n']))
            mesh = [a * i / (n - i) for i in indices]
        elif eq == 'r=a*(exp(d*i)-1)':
            a, d = (float(grid_params['a']), float(grid_params['d']))
            mesh = [a * (np.exp(d * i) - 1.0) for i in indices]
        elif eq == 'r=d*i':
            d = float(grid_params['d'])
            mesh = [d * i for i in indices]
        elif eq == 'r=(i/n+a)^5/a-a^4':
            a, n = (float(grid_params['a']), float(grid_params['n']))
            mesh = [(i / n + a) ** 5 / a - a ** 4 for i in indices]
        else:
            raise ValueError(f'Unknown grid type: {eq}')
        return np.array(mesh)

    def _parse_radfunc(self, func_name):
        """Parse the first occurrence of func_name in the XML file."""
        node = self.root.find(func_name)
        grid = node.attrib['grid']
        values = np.array([float(s) for s in node.text.split()])
        return (self.rad_grids[grid], values, node.attrib)

    def _parse_all_radfuncs(self, func_name):
        """Parse all the nodes with tag func_name in the XML file."""
        for node in self.root.findall(func_name):
            grid = node.attrib['grid']
            values = np.array([float(s) for s in node.text.split()])
            yield (self.rad_grids[grid], values, node.attrib)

    @lazy_property
    def ae_core_density(self):
        """The all-electron radial density."""
        mesh, values, _attrib = self._parse_radfunc('ae_core_density')
        return RadialFunction(mesh, values)

    @lazy_property
    def pseudo_core_density(self):
        """The pseudized radial density."""
        mesh, values, _attrib = self._parse_radfunc('pseudo_core_density')
        return RadialFunction(mesh, values)

    @lazy_property
    def ae_partial_waves(self):
        """Dictionary with the AE partial waves indexed by state."""
        ae_partial_waves = {}
        for mesh, values, attrib in self._parse_all_radfuncs('ae_partial_wave'):
            state = attrib['state']
            ae_partial_waves[state] = RadialFunction(mesh, values)
        return ae_partial_waves

    @property
    def pseudo_partial_waves(self):
        """Dictionary with the pseudo partial waves indexed by state."""
        pseudo_partial_waves = {}
        for mesh, values, attrib in self._parse_all_radfuncs('pseudo_partial_wave'):
            state = attrib['state']
            pseudo_partial_waves[state] = RadialFunction(mesh, values)
        return pseudo_partial_waves

    @lazy_property
    def projector_functions(self):
        """Dictionary with the PAW projectors indexed by state."""
        projector_functions = {}
        for mesh, values, attrib in self._parse_all_radfuncs('projector_function'):
            state = attrib['state']
            projector_functions[state] = RadialFunction(mesh, values)
        return projector_functions

    def yield_figs(self, **kwargs):
        """This function *generates* a predefined list of matplotlib figures with minimal input from the user."""
        yield self.plot_densities(title='PAW densities', show=False)
        yield self.plot_waves(title='PAW waves', show=False)
        yield self.plot_projectors(title='PAW projectors', show=False)

    @add_fig_kwargs
    def plot_densities(self, ax: plt.Axes=None, **kwargs):
        """
        Plot the PAW densities.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            `matplotlib` figure
        """
        ax, fig = get_ax_fig(ax)
        ax.grid(visible=True)
        ax.set_xlabel('r [Bohr]')
        for idx, density_name in enumerate(['ae_core_density', 'pseudo_core_density']):
            rden = getattr(self, density_name)
            label = '$n_c$' if idx == 1 else '$\\tilde{n}_c$'
            ax.plot(rden.mesh, rden.mesh * rden.values, label=label, lw=2)
        ax.legend(loc='best')
        return fig

    @add_fig_kwargs
    def plot_waves(self, ax: plt.Axes=None, fontsize=12, **kwargs):
        """
        Plot the AE and the pseudo partial waves.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.
            fontsize: fontsize for legends and titles

        Returns:
            plt.Figure: matplotlib figure
        """
        ax, fig = get_ax_fig(ax)
        ax.grid(visible=True)
        ax.set_xlabel('r [Bohr]')
        ax.set_ylabel('$r\\phi,\\, r\\tilde\\phi\\, [Bohr]^{-\\frac{1}{2}}$')
        for state, rfunc in self.pseudo_partial_waves.items():
            ax.plot(rfunc.mesh, rfunc.mesh * rfunc.values, lw=2, label='PS-WAVE: ' + state)
        for state, rfunc in self.ae_partial_waves.items():
            ax.plot(rfunc.mesh, rfunc.mesh * rfunc.values, lw=2, label='AE-WAVE: ' + state)
        ax.legend(loc='best', shadow=True, fontsize=fontsize)
        return fig

    @add_fig_kwargs
    def plot_projectors(self, ax: plt.Axes=None, fontsize=12, **kwargs):
        """
        Plot the PAW projectors.

        Args:
            ax: matplotlib Axes or None if a new figure should be created.

        Returns:
            plt.Figure: matplotlib figure
        """
        ax, fig = get_ax_fig(ax)
        ax.grid(visible=True)
        ax.set_xlabel('r [Bohr]')
        ax.set_ylabel('$r\\tilde p\\, [Bohr]^{-\\frac{1}{2}}$')
        for state, rfunc in self.projector_functions.items():
            ax.plot(rfunc.mesh, rfunc.mesh * rfunc.values, label='TPROJ: ' + state)
        ax.legend(loc='best', shadow=True, fontsize=fontsize)
        return fig