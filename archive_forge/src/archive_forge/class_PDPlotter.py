from __future__ import annotations
import collections
import itertools
import json
import logging
import math
import os
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, no_type_check
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import FontProperties
from monty.json import MontyDecoder, MSONable
from scipy import interpolate
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from tqdm import tqdm
from pymatgen.analysis.reaction_calculator import Reaction, ReactionError
from pymatgen.core import DummySpecies, Element, get_el_sp
from pymatgen.core.composition import Composition
from pymatgen.entries import Entry
from pymatgen.util.coord import Simplex, in_coord_list
from pymatgen.util.due import Doi, due
from pymatgen.util.plotting import pretty_plot
from pymatgen.util.string import htmlify, latexify
class PDPlotter:
    """
    A plotting class for compositional phase diagrams.

    To use, initialize this class with a PhaseDiagram object containing 1-4 components
    and call get_plot() or show().
    """

    def __init__(self, phasediagram: PhaseDiagram, show_unstable: float=0.2, backend: Literal['plotly', 'matplotlib']='plotly', ternary_style: Literal['2d', '3d']='2d', **plotkwargs):
        """
        Args:
            phasediagram (PhaseDiagram): PhaseDiagram object (must be 1-4 components).
            show_unstable (float): Whether unstable (above the hull) phases will be
                plotted. If a number > 0 is entered, all phases with
                e_hull < show_unstable (eV/atom) will be shown.
            backend ("plotly" | "matplotlib"): Python package to use for plotting.
                Defaults to "plotly".
            ternary_style ("2d" | "3d"): Ternary phase diagrams are typically plotted in
                two-dimensions (2d), but can be plotted in three dimensions (3d) to visualize
                the depth of the hull. This argument only applies when backend="plotly".
                Defaults to "2d".
            **plotkwargs (dict): Keyword args passed to matplotlib.pyplot.plot (only
                applies when backend="matplotlib"). Can be used to customize markers
                etc. If not set, the default is:
                    {
                        "markerfacecolor": "#4daf4a",
                        "markersize": 10,
                        "linewidth": 3
                    }.
        """
        dim = len(phasediagram.elements)
        if dim >= 5:
            raise ValueError('Only 1-4 components supported!')
        self._pd = phasediagram
        self.show_unstable = show_unstable
        self.backend = backend
        self.ternary_style = ternary_style.lower()
        self.lines = uniquelines(self._pd.facets) if dim > 1 else [[self._pd.facets[0][0], self._pd.facets[0][0]]]
        self._min_energy = min((self._pd.get_form_energy_per_atom(entry) for entry in self._pd.stable_entries))
        self._dim = dim
        self.plotkwargs = plotkwargs or {'markerfacecolor': '#4daf4a', 'markersize': 10, 'linewidth': 3}

    def get_plot(self, label_stable: bool=True, label_unstable: bool=True, ordering: Sequence[str] | None=None, energy_colormap=None, process_attributes: bool=False, ax: plt.Axes=None, label_uncertainties: bool=False, fill: bool=True, highlight_entries: Collection[PDEntry] | None=None) -> go.Figure | plt.Axes:
        """
        Args:
            label_stable: Whether to label stable compounds.
            label_unstable: Whether to label unstable compounds.
            ordering: Ordering of vertices, given as a list ['Up',
                'Left','Right'] (matplotlib only).
            energy_colormap: Colormap for coloring energy (matplotlib only).
            process_attributes: Whether to process the attributes (matplotlib only).
            ax: Existing matplotlib Axes object if plotting multiple phase diagrams
                (matplotlib only).
            label_uncertainties: Whether to add error bars to the hull.
                For binaries, this also shades the hull with the uncertainty window.
                (plotly only).
            fill: Whether to shade the hull. For ternary_2d and quaternary plots, this
                colors facets arbitrarily for visual clarity. For ternary_3d plots, this
                shades the hull by formation energy (plotly only).
            highlight_entries: Entries to highlight in the plot (plotly only). This will
                create a new marker trace that is separate from the other entries.

        Returns:
            go.Figure | plt.Axes: Plotly figure or matplotlib axes object depending on backend.
        """
        fig = None
        data = []
        if self.backend == 'plotly':
            if self._dim != 1:
                data.append(self._create_plotly_lines())
            stable_marker_plot, unstable_marker_plot, highlight_plot = self._create_plotly_markers(highlight_entries, label_uncertainties)
            if self._dim == 2 and label_uncertainties:
                data.append(self._create_plotly_uncertainty_shading(stable_marker_plot))
            if self._dim == 3 and self.ternary_style == '3d':
                data.append(self._create_plotly_ternary_support_lines())
            if self._dim != 1 and (not (self._dim == 3 and self.ternary_style == '2d')):
                data.append(self._create_plotly_stable_labels(label_stable))
            if fill and self._dim in [3, 4]:
                data.extend(self._create_plotly_fill())
            data.extend([stable_marker_plot, unstable_marker_plot])
            if highlight_plot is not None:
                data.append(highlight_plot)
            fig = go.Figure(data=data)
            fig.layout = self._create_plotly_figure_layout()
            fig.update_layout(coloraxis_colorbar={'yanchor': 'top', 'y': 0.05, 'x': 1})
        elif self.backend == 'matplotlib':
            if self._dim <= 3:
                fig = self._get_matplotlib_2d_plot(label_stable, label_unstable, ordering, energy_colormap, ax=ax, process_attributes=process_attributes)
            elif self._dim == 4:
                fig = self._get_matplotlib_3d_plot(label_stable, ax=ax)
        return fig

    def show(self, *args, **kwargs) -> None:
        """
        Draw the phase diagram with the provided arguments and display it. This shows
        the figure but does not return it.

        Args:
            *args: Passed to get_plot.
            **kwargs: Passed to get_plot.
        """
        plot = self.get_plot(*args, **kwargs)
        if self.backend == 'matplotlib':
            plot.get_figure().show()
        else:
            plot.show()

    def write_image(self, stream: str | StringIO, image_format: str='svg', **kwargs) -> None:
        """
        Directly save the plot to a file. This is a wrapper for calling plt.savefig() or
        fig.write_image(), depending on the backend. For more customization, it is
        recommended to call those methods directly.

        Args:
            stream (str | StringIO): Filename or StringIO stream.
            image_format (str): Can be any supported image format for the plotting backend.
                Defaults to 'svg' (vector graphics).
            **kwargs: Optinoal kwargs passed to the get_plot function.
        """
        if self.backend == 'matplotlib':
            ax = self.get_plot(**kwargs)
            ax.figure.set_size_inches((12, 10))
            ax.figure.savefig(stream, format=image_format)
        elif self.backend == 'plotly':
            fig = self.get_plot(**kwargs)
            fig.write_image(stream, format=image_format)

    def plot_element_profile(self, element, comp, show_label_index=None, xlim=5):
        """
        Draw the element profile plot for a composition varying different
        chemical potential of an element.

        X value is the negative value of the chemical potential reference to
        elemental chemical potential. For example, if choose Element("Li"),
        X= -(µLi-µLi0), which corresponds to the voltage versus metal anode.
        Y values represent for the number of element uptake in this composition
        (unit: per atom). All reactions are printed to help choosing the
        profile steps you want to show label in the plot.

        Args:
            element (Element): An element of which the chemical potential is
                considered. It also must be in the phase diagram.
            comp (Composition): A composition.
            show_label_index (list of integers): The labels for reaction products
                you want to show in the plot. Default to None (not showing any
                annotation for reaction products). For the profile steps you want
                to show the labels, just add it to the show_label_index. The
                profile step counts from zero. For example, you can set
                show_label_index=[0, 2, 5] to label profile step 0,2,5.
            xlim (float): The max x value. x value is from 0 to xlim. Default to
                5 eV.

        Returns:
            Plot of element profile evolution by varying the chemical potential
            of an element.
        """
        ax = pretty_plot(12, 8)
        pd = self._pd
        evolution = pd.get_element_profile(element, comp)
        num_atoms = evolution[0]['reaction'].reactants[0].num_atoms
        element_energy = evolution[0]['chempot']
        x1, x2, y1 = (None, None, None)
        for idx, dct in enumerate(evolution):
            v = -(dct['chempot'] - element_energy)
            if idx != 0:
                ax.plot([x2, x2], [y1, dct['evolution'] / num_atoms], 'k', linewidth=2.5)
            x1 = v
            y1 = dct['evolution'] / num_atoms
            x2 = -(evolution[idx + 1]['chempot'] - element_energy) if idx != len(evolution) - 1 else 5.0
            if show_label_index is not None and idx in show_label_index:
                products = [re.sub('(\\d+)', '$_{\\1}$', p.reduced_formula) for p in dct['reaction'].products if p.reduced_formula != element.symbol]
                ax.annotate(', '.join(products), xy=(v + 0.05, y1 + 0.05), fontsize=24, color='r')
                ax.plot([x1, x2], [y1, y1], 'r', linewidth=3)
            else:
                ax.plot([x1, x2], [y1, y1], 'k', linewidth=2.5)
        ax.set_xlim((0, xlim))
        ax.set_xlabel('-$\\Delta{\\mu}$ (eV)')
        ax.set_ylabel('Uptake per atom')
        return ax

    def plot_chempot_range_map(self, elements, referenced=True) -> None:
        """
        Plot the chemical potential range _map using matplotlib. Currently works only for
        3-component PDs. This shows the plot but does not return it.

        Note: this functionality is now included in the ChemicalPotentialDiagram
        class (pymatgen.analysis.chempot_diagram).

        Args:
            elements: Sequence of elements to be considered as independent
                variables. E.g., if you want to show the stability ranges of
                all Li-Co-O phases w.r.t. to uLi and uO, you will supply
                [Element("Li"), Element("O")]
            referenced: if True, gives the results with a reference being the
                        energy of the elemental phase. If False, gives absolute values.
        """
        self.get_chempot_range_map_plot(elements, referenced=referenced).show()

    def get_chempot_range_map_plot(self, elements, referenced=True):
        """
        Returns a plot of the chemical potential range _map. Currently works
        only for 3-component PDs.

        Note: this functionality is now included in the ChemicalPotentialDiagram
        class (pymatgen.analysis.chempot_diagram).

        Args:
            elements: Sequence of elements to be considered as independent
                variables. E.g., if you want to show the stability ranges of
                all Li-Co-O phases w.r.t. to uLi and uO, you will supply
                [Element("Li"), Element("O")]
            referenced: if True, gives the results with a reference being the
                energy of the elemental phase. If False, gives absolute values.

        Returns:
            plt.Axes: matplotlib axes object.
        """
        ax = pretty_plot(12, 8)
        chempot_ranges = self._pd.get_chempot_range_map(elements, referenced=referenced)
        missing_lines = {}
        excluded_region = []
        for entry, lines in chempot_ranges.items():
            comp = entry.composition
            center_x = 0
            center_y = 0
            coords = []
            contain_zero = any((comp.get_atomic_fraction(el) == 0 for el in elements))
            is_boundary = not contain_zero and sum((comp.get_atomic_fraction(el) for el in elements)) == 1
            for line in lines:
                x, y = line.coords.transpose()
                plt.plot(x, y, 'k-')
                for coord in line.coords:
                    if not in_coord_list(coords, coord):
                        coords.append(coord.tolist())
                        center_x += coord[0]
                        center_y += coord[1]
                if is_boundary:
                    excluded_region.extend(line.coords)
            if coords and contain_zero:
                missing_lines[entry] = coords
            else:
                xy = (center_x / len(coords), center_y / len(coords))
                plt.annotate(latexify(entry.name), xy, fontsize=22)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        excluded_region.append([xlim[1], ylim[1]])
        excluded_region = sorted(excluded_region, key=lambda c: c[0])
        x, y = np.transpose(excluded_region)
        plt.fill(x, y, '0.80')
        el0 = elements[0]
        el1 = elements[1]
        for entry, coords in missing_lines.items():
            center_x = sum((c[0] for c in coords))
            center_y = sum((c[1] for c in coords))
            comp = entry.composition
            is_x = comp.get_atomic_fraction(el0) < 0.01
            is_y = comp.get_atomic_fraction(el1) < 0.01
            n = len(coords)
            if not (is_x and is_y):
                if is_x:
                    coords = sorted(coords, key=lambda c: c[1])
                    for idx in [0, -1]:
                        x = [min(xlim), coords[idx][0]]
                        y = [coords[idx][1], coords[idx][1]]
                        plt.plot(x, y, 'k')
                        center_x += min(xlim)
                        center_y += coords[idx][1]
                elif is_y:
                    coords = sorted(coords, key=lambda c: c[0])
                    for idx in [0, -1]:
                        x = [coords[idx][0], coords[idx][0]]
                        y = [coords[idx][1], min(ylim)]
                        plt.plot(x, y, 'k')
                        center_x += coords[idx][0]
                        center_y += min(ylim)
                xy = (center_x / (n + 2), center_y / (n + 2))
            else:
                center_x = sum((coord[0] for coord in coords)) + xlim[0]
                center_y = sum((coord[1] for coord in coords)) + ylim[0]
                xy = (center_x / (n + 1), center_y / (n + 1))
            ax.annotate(latexify(entry.name), xy, horizontalalignment='center', verticalalignment='center', fontsize=22)
        ax.set_xlabel(f'$\\mu_{{{el0.symbol}}} - \\mu_{{{el0.symbol}}}^0$ (eV)')
        ax.set_ylabel(f'$\\mu_{{{el1.symbol}}} - \\mu_{{{el1.symbol}}}^0$ (eV)')
        plt.tight_layout()
        return ax

    def get_contour_pd_plot(self):
        """
        Plot a contour phase diagram plot, where phase triangles are colored
        according to degree of instability by interpolation. Currently only
        works for 3-component phase diagrams.

        Returns:
            A matplotlib plot object.
        """
        pd = self._pd
        entries = pd.qhull_entries
        data = np.array(pd.qhull_data)
        ax = self._get_matplotlib_2d_plot()
        data[:, 0:2] = triangular_coord(data[:, 0:2]).transpose()
        for idx, entry in enumerate(entries):
            data[idx, 2] = self._pd.get_e_above_hull(entry)
        gridsize = 0.005
        xnew = np.arange(0, 1.0, gridsize)
        ynew = np.arange(0, 1, gridsize)
        f = interpolate.LinearNDInterpolator(data[:, 0:2], data[:, 2])
        znew = np.zeros((len(ynew), len(xnew)))
        for idx, xval in enumerate(xnew):
            for j, yval in enumerate(ynew):
                znew[j, idx] = f(xval, yval)
        contourf = ax.contourf(xnew, ynew, znew, 1000, cmap=cm.autumn_r)
        plt.colorbar(contourf)
        return ax

    @property
    @lru_cache(1)
    def pd_plot_data(self):
        """
        Plotting data for phase diagram. Cached for repetitive calls.

        2-comp - Full hull with energies
        3/4-comp - Projection into 2D or 3D Gibbs triangles

        Returns:
            A tuple containing three objects (lines, stable_entries, unstable_entries):
            - lines is a list of list of coordinates for lines in the PD.
            - stable_entries is a dict of {coordinates : entry} for each stable node
                in the phase diagram. (Each coordinate can only have one
                stable phase)
            - unstable_entries is a dict of {entry: coordinates} for all unstable
                nodes in the phase diagram.
        """
        pd = self._pd
        entries = pd.qhull_entries
        data = np.array(pd.qhull_data)
        lines = []
        stable_entries = {}
        for line in self.lines:
            entry1 = entries[line[0]]
            entry2 = entries[line[1]]
            if self._dim < 3:
                x = [data[line[0]][0], data[line[1]][0]]
                y = [pd.get_form_energy_per_atom(entry1), pd.get_form_energy_per_atom(entry2)]
                coord = [x, y]
            elif self._dim == 3:
                coord = triangular_coord(data[line, 0:2])
            else:
                coord = tet_coord(data[line, 0:3])
            lines.append(coord)
            labelcoord = list(zip(*coord))
            stable_entries[labelcoord[0]] = entry1
            stable_entries[labelcoord[1]] = entry2
        all_entries = pd.all_entries
        all_data = np.array(pd.all_entries_hulldata)
        unstable_entries = {}
        stable = pd.stable_entries
        for idx, entry in enumerate(all_entries):
            if entry not in stable:
                if self._dim < 3:
                    x = [all_data[idx][0], all_data[idx][0]]
                    y = [pd.get_form_energy_per_atom(entry), pd.get_form_energy_per_atom(entry)]
                    coord = [x, y]
                elif self._dim == 3:
                    coord = triangular_coord([all_data[idx, 0:2], all_data[idx, 0:2]])
                else:
                    coord = tet_coord([all_data[idx, 0:3], all_data[idx, 0:3], all_data[idx, 0:3]])
                labelcoord = list(zip(*coord))
                unstable_entries[entry] = labelcoord[0]
        return (lines, stable_entries, unstable_entries)

    def _create_plotly_figure_layout(self, label_stable=True):
        """
        Creates layout for plotly phase diagram figure and updates with
        figure annotations.

        Args:
            label_stable (bool): Whether to label stable compounds

        Returns:
            Dictionary with Plotly figure layout settings.
        """
        annotations_list = None
        layout = {}
        if label_stable:
            annotations_list = self._create_plotly_element_annotations()
        if self._dim == 1:
            layout = plotly_layouts['default_unary_layout'].copy()
        if self._dim == 2:
            layout = plotly_layouts['default_binary_layout'].copy()
            layout['xaxis']['title'] = f'Composition (Fraction {self._pd.elements[1]})'
            layout['annotations'] = annotations_list
        elif self._dim == 3 and self.ternary_style == '2d':
            layout = plotly_layouts['default_ternary_2d_layout'].copy()
            for el, axis in zip(self._pd.elements, ['a', 'b', 'c']):
                el_ref = self._pd.el_refs[el]
                clean_formula = str(el_ref.elements[0])
                if hasattr(el_ref, 'original_entry'):
                    clean_formula = htmlify(el_ref.original_entry.reduced_formula)
                layout['ternary'][axis + 'axis']['title'] = {'text': clean_formula, 'font': {'size': 24}}
        elif self._dim == 3 and self.ternary_style == '3d':
            layout = plotly_layouts['default_ternary_3d_layout'].copy()
            layout['scene']['annotations'] = annotations_list
        elif self._dim == 4:
            layout = plotly_layouts['default_quaternary_layout'].copy()
            layout['scene']['annotations'] = annotations_list
        return layout

    def _create_plotly_lines(self):
        """
        Create Plotly scatter plots containing line traces of phase diagram facets.

        Returns:
            Either a go.Scatter (binary), go.Scatterternary (ternary_2d), or
            go.Scatter3d plot (ternary_3d, quaternary)
        """
        line_plot = None
        x, y, z, energies = ([], [], [], [])
        pd = self._pd
        plot_args = {'mode': 'lines', 'hoverinfo': 'none', 'line': {'color': 'black', 'width': 4.0}, 'showlegend': False}
        if self._dim == 3 and self.ternary_style == '2d':
            plot_args['line']['width'] = 1.5
            el_a, el_b, el_c = pd.elements
            for line in uniquelines(pd.facets):
                e0 = pd.qhull_entries[line[0]]
                e1 = pd.qhull_entries[line[1]]
                x += [e0.composition[el_a], e1.composition[el_a], None]
                y += [e0.composition[el_b], e1.composition[el_b], None]
                z += [e0.composition[el_c], e1.composition[el_c], None]
        else:
            for line in self.pd_plot_data[0]:
                x += [*line[0], None]
                y += [*line[1], None]
                if self._dim == 3:
                    form_enes = [self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord]) for coord in zip(line[0], line[1])]
                    z += [*form_enes, None]
                elif self._dim == 4:
                    form_enes = [self._pd.get_form_energy_per_atom(self.pd_plot_data[1][coord]) for coord in zip(line[0], line[1], line[2])]
                    energies += [*form_enes, None]
                    z += [*line[2], None]
        if self._dim == 2:
            line_plot = go.Scatter(x=x, y=y, **plot_args)
        elif self._dim == 3 and self.ternary_style == '2d':
            line_plot = go.Scatterternary(a=x, b=y, c=z, **plot_args)
        elif self._dim == 3 and self.ternary_style == '3d':
            line_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
        elif self._dim == 4:
            plot_args['line']['width'] = 1.5
            line_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)
        return line_plot

    def _create_plotly_fill(self):
        """
        Creates shaded mesh traces for coloring the hull.

        For tenrary_3d plots, the color shading is based on formation energy.

        Returns:
            go.Mesh3d plot
        """
        traces = []
        pd = self._pd
        if self._dim == 3 and self.ternary_style == '2d':
            fillcolors = itertools.cycle(plotly_layouts['default_fill_colors'])
            el_a, el_b, el_c = pd.elements
            for _idx, facet in enumerate(pd.facets):
                a = []
                b = []
                c = []
                e0, e1, e2 = sorted((pd.qhull_entries[facet[idx]] for idx in range(3)), key=lambda x: x.reduced_formula)
                a = [e0.composition[el_a], e1.composition[el_a], e2.composition[el_a]]
                b = [e0.composition[el_b], e1.composition[el_b], e2.composition[el_b]]
                c = [e0.composition[el_c], e1.composition[el_c], e2.composition[el_c]]
                e_strs = []
                for entry in (e0, e1, e2):
                    if hasattr(entry, 'original_entry'):
                        entry = entry.original_entry
                    e_strs.append(htmlify(entry.reduced_formula))
                name = f'{e_strs[0]}—{e_strs[1]}—{e_strs[2]}'
                traces += [go.Scatterternary(a=a, b=b, c=c, mode='lines', fill='toself', line={'width': 0}, fillcolor=next(fillcolors), opacity=0.15, hovertemplate='<extra></extra>', name=name, showlegend=False)]
        elif self._dim == 3 and self.ternary_style == '3d':
            facets = np.array(self._pd.facets)
            coords = np.array([triangular_coord(c) for c in zip(self._pd.qhull_data[:-1, 0], self._pd.qhull_data[:-1, 1])])
            energies = np.array([self._pd.get_form_energy_per_atom(entry) for entry in self._pd.qhull_entries])
            traces.append(go.Mesh3d(x=list(coords[:, 1]), y=list(coords[:, 0]), z=list(energies), i=list(facets[:, 1]), j=list(facets[:, 0]), k=list(facets[:, 2]), opacity=0.7, intensity=list(energies), colorscale=plotly_layouts['stable_colorscale'], colorbar={'title': 'Formation energy<br>(eV/atom)', 'x': 0.9, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}, hoverinfo='none', lighting={'diffuse': 0.0, 'ambient': 1.0}, name='Convex Hull (shading)', flatshading=True, showlegend=True))
        elif self._dim == 4:
            all_data = np.array(pd.qhull_data)
            fillcolors = itertools.cycle(plotly_layouts['default_fill_colors'])
            for _idx, facet in enumerate(pd.facets):
                xs, ys, zs = ([], [], [])
                for v in facet:
                    x, y, z = tet_coord(all_data[v, 0:3])
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                traces += [go.Mesh3d(x=xs, y=ys, z=zs, opacity=0.05, alphahull=-1, flatshading=True, hoverinfo='skip', color=next(fillcolors))]
        return traces

    def _create_plotly_stable_labels(self, label_stable=True):
        """
        Creates a (hidable) scatter trace containing labels of stable phases.
        Contains some functionality for creating sensible label positions. This method
        does not apply to 2D ternary plots (stable labels are turned off).

        Returns:
            go.Scatter (or go.Scatter3d) plot
        """
        x, y, z, text, textpositions = ([], [], [], [], [])
        stable_labels_plot = min_energy_x = None
        offset_2d = 0.008
        offset_3d = 0.01
        energy_offset = -0.05 * self._min_energy
        if self._dim == 2:
            min_energy_x = min(list(self.pd_plot_data[1]), key=lambda c: c[1])[0]
        for coords, entry in self.pd_plot_data[1].items():
            if entry.composition.is_element:
                continue
            x_coord = coords[0]
            y_coord = coords[1]
            textposition = None
            if self._dim == 2:
                textposition = 'bottom left'
                if x_coord >= min_energy_x:
                    textposition = 'bottom right'
                    x_coord += offset_2d
                else:
                    x_coord -= offset_2d
                y_coord -= offset_2d + 0.005
            elif self._dim == 3 and self.ternary_style == '3d':
                textposition = 'middle center'
                if coords[0] > 0.5:
                    x_coord += offset_3d
                else:
                    x_coord -= offset_3d
                if coords[1] > 0.866 / 2:
                    y_coord -= offset_3d
                else:
                    y_coord += offset_3d
                z.append(self._pd.get_form_energy_per_atom(entry) + energy_offset)
            elif self._dim == 4:
                x_coord = x_coord - offset_3d
                y_coord = y_coord - offset_3d
                textposition = 'bottom right'
                z.append(coords[2])
            x.append(x_coord)
            y.append(y_coord)
            textpositions.append(textposition)
            comp = entry.composition
            if hasattr(entry, 'original_entry'):
                comp = entry.original_entry.composition
            formula = comp.reduced_formula
            text.append(htmlify(formula))
        visible = True
        if not label_stable or self._dim == 4:
            visible = 'legendonly'
        plot_args = {'text': text, 'textposition': textpositions, 'mode': 'text', 'name': 'Labels (stable)', 'hoverinfo': 'skip', 'opacity': 1.0, 'visible': visible, 'showlegend': True}
        if self._dim == 2:
            stable_labels_plot = go.Scatter(x=x, y=y, **plot_args)
        elif self._dim == 3 and self.ternary_style == '3d':
            stable_labels_plot = go.Scatter3d(x=y, y=x, z=z, **plot_args)
        elif self._dim == 4:
            stable_labels_plot = go.Scatter3d(x=x, y=y, z=z, **plot_args)
        return stable_labels_plot

    def _create_plotly_element_annotations(self):
        """
        Creates terminal element annotations for Plotly phase diagrams. This method does
        not apply to ternary_2d plots.

        Functionality is included for phase diagrams with non-elemental endmembers
        (as is true for grand potential phase diagrams).

        Returns:
            List of annotation dicts.
        """
        annotations_list = []
        x, y, z = (None, None, None)
        if self._dim == 3 and self.ternary_style == '2d':
            return None
        for coords, entry in self.pd_plot_data[1].items():
            if not entry.composition.is_element:
                continue
            x, y = (coords[0], coords[1])
            if self._dim == 3:
                z = self._pd.get_form_energy_per_atom(entry)
            elif self._dim == 4:
                z = coords[2]
            if entry.composition.is_element:
                clean_formula = str(entry.elements[0])
                if hasattr(entry, 'original_entry'):
                    orig_comp = entry.original_entry.composition
                    clean_formula = htmlify(orig_comp.reduced_formula)
                font_dict = {'color': '#000000', 'size': 24.0}
                opacity = 1.0
            offset = 0.03 if self._dim == 2 else 0.06
            if x < 0.4:
                x -= offset
            elif x > 0.6:
                x += offset
            if y < 0.1:
                y -= offset
            elif y > 0.8:
                y += offset
            if self._dim == 4 and z > 0.8:
                z += offset
            annotation = plotly_layouts['default_annotation_layout'].copy()
            annotation.update(x=x, y=y, font=font_dict, text=clean_formula, opacity=opacity)
            if self._dim in (3, 4):
                for d in ['xref', 'yref']:
                    annotation.pop(d)
                    if self._dim == 3:
                        annotation.update({'x': y, 'y': x})
                        if entry.composition.is_element:
                            z = 0.9 * self._min_energy
                annotation['z'] = z
            annotations_list.append(annotation)
        if self._dim == 3:
            annotations_list.append({'x': 1, 'y': 1, 'z': 0, 'opacity': 0, 'text': ''})
        return annotations_list

    def _create_plotly_markers(self, highlight_entries=None, label_uncertainties=False):
        """
        Creates stable and unstable marker plots for overlaying on the phase diagram.

        Returns:
            Tuple of Plotly go.Scatter (unary, binary), go.Scatterternary(ternary_2d),
            or go.Scatter3d (ternary_3d, quaternary) objects in order:
            (stable markers, unstable markers)
        """

        def get_marker_props(coords, entries):
            """Method for getting marker locations, hovertext, and error bars
            from pd_plot_data.
            """
            x, y, z, texts, energies, uncertainties = ([], [], [], [], [], [])
            is_stable = [entry in self._pd.stable_entries for entry in entries]
            for coord, entry, stable in zip(coords, entries, is_stable):
                energy = round(self._pd.get_form_energy_per_atom(entry), 3)
                entry_id = getattr(entry, 'entry_id', 'no ID')
                comp = entry.composition
                if hasattr(entry, 'original_entry'):
                    orig_entry = entry.original_entry
                    comp = orig_entry.composition
                    entry_id = getattr(orig_entry, 'entry_id', 'no ID')
                formula = comp.reduced_formula
                clean_formula = htmlify(formula)
                label = f'{clean_formula} ({entry_id}) <br> {energy} eV/atom'
                if not stable:
                    e_above_hull = round(self._pd.get_e_above_hull(entry), 3)
                    if e_above_hull > self.show_unstable:
                        continue
                    label += f' ({e_above_hull:+} eV/atom)'
                    energies.append(e_above_hull)
                else:
                    uncertainty = 0
                    label += ' (Stable)'
                    if hasattr(entry, 'correction_uncertainty_per_atom') and label_uncertainties:
                        uncertainty = round(entry.correction_uncertainty_per_atom, 4)
                        label += f'<br> (Error: +/- {uncertainty} eV/atom)'
                    uncertainties.append(uncertainty)
                    energies.append(energy)
                texts.append(label)
                if self._dim == 3 and self.ternary_style == '2d':
                    for el, axis in zip(self._pd.elements, [x, y, z]):
                        axis.append(entry.composition[el])
                else:
                    x.append(coord[0])
                    y.append(coord[1])
                    if self._dim == 3:
                        z.append(energy)
                    elif self._dim == 4:
                        z.append(coord[2])
            return {'x': x, 'y': y, 'z': z, 'texts': texts, 'energies': energies, 'uncertainties': uncertainties}
        if highlight_entries is None:
            highlight_entries = []
        stable_coords, stable_entries = ([], [])
        unstable_coords, unstable_entries = ([], [])
        highlight_coords, highlight_ents = ([], [])
        for coord, entry in zip(self.pd_plot_data[1], self.pd_plot_data[1].values()):
            if entry in highlight_entries:
                highlight_coords.append(coord)
                highlight_ents.append(entry)
            else:
                stable_coords.append(coord)
                stable_entries.append(entry)
        for coord, entry in zip(self.pd_plot_data[2].values(), self.pd_plot_data[2]):
            if entry in highlight_entries:
                highlight_coords.append(coord)
                highlight_ents.append(entry)
            else:
                unstable_coords.append(coord)
                unstable_entries.append(entry)
        stable_props = get_marker_props(stable_coords, stable_entries)
        unstable_props = get_marker_props(unstable_coords, unstable_entries)
        highlight_props = get_marker_props(highlight_coords, highlight_entries)
        stable_markers, unstable_markers, highlight_markers = ({}, {}, {})
        if self._dim == 1:
            stable_markers = plotly_layouts['default_unary_marker_settings'].copy()
            unstable_markers = plotly_layouts['default_unary_marker_settings'].copy()
            stable_markers.update(x=[0] * len(stable_props['y']), y=list(stable_props['x']), name='Stable', marker={'color': 'darkgreen', 'size': 20, 'line': {'color': 'black', 'width': 2}, 'symbol': 'star'}, opacity=0.9, hovertext=stable_props['texts'], error_y={'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
            plotly_layouts['unstable_colorscale'].copy()
            unstable_markers.update(x=[0] * len(unstable_props['y']), y=list(unstable_props['x']), name='Above Hull', marker={'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 16, 'symbol': 'diamond-wide', 'line': {'color': 'black', 'width': 2}}, hovertext=unstable_props['texts'], opacity=0.9)
            if highlight_entries:
                highlight_markers = plotly_layouts['default_unary_marker_settings'].copy()
                highlight_markers.update({'x': [0] * len(highlight_props['y']), 'y': list(highlight_props['x']), 'name': 'Highlighted', 'marker': {'color': 'mediumvioletred', 'size': 22, 'line': {'color': 'black', 'width': 2}, 'symbol': 'square'}, 'opacity': 0.9, 'hovertext': highlight_props['texts'], 'error_y': {'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5}})
        if self._dim == 2:
            stable_markers = plotly_layouts['default_binary_marker_settings'].copy()
            unstable_markers = plotly_layouts['default_binary_marker_settings'].copy()
            stable_markers.update(x=list(stable_props['x']), y=list(stable_props['y']), name='Stable', marker={'color': 'darkgreen', 'size': 16, 'line': {'color': 'black', 'width': 2}}, opacity=0.99, hovertext=stable_props['texts'], error_y={'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
            unstable_markers.update({'x': list(unstable_props['x']), 'y': list(unstable_props['y']), 'name': 'Above Hull', 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 7, 'symbol': 'diamond', 'line': {'color': 'black', 'width': 1}, 'opacity': 0.8}, 'hovertext': unstable_props['texts']})
            if highlight_entries:
                highlight_markers = plotly_layouts['default_binary_marker_settings'].copy()
                highlight_markers.update(x=list(highlight_props['x']), y=list(highlight_props['y']), name='Highlighted', marker={'color': 'mediumvioletred', 'size': 16, 'line': {'color': 'black', 'width': 2}, 'symbol': 'square'}, opacity=0.99, hovertext=highlight_props['texts'], error_y={'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'gray', 'thickness': 2.5, 'width': 5})
        elif self._dim == 3 and self.ternary_style == '2d':
            stable_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
            unstable_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
            stable_markers.update({'a': list(stable_props['x']), 'b': list(stable_props['y']), 'c': list(stable_props['z']), 'name': 'Stable', 'hovertext': stable_props['texts'], 'marker': {'color': 'green', 'line': {'width': 2.0, 'color': 'black'}, 'symbol': 'circle', 'size': 15}})
            unstable_markers.update({'a': unstable_props['x'], 'b': unstable_props['y'], 'c': unstable_props['z'], 'name': 'Above Hull', 'hovertext': unstable_props['texts'], 'marker': {'color': unstable_props['energies'], 'opacity': 0.8, 'colorscale': plotly_layouts['unstable_colorscale'], 'line': {'width': 1, 'color': 'black'}, 'size': 7, 'symbol': 'diamond', 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}})
            if highlight_entries:
                highlight_markers = plotly_layouts['default_ternary_2d_marker_settings'].copy()
                highlight_markers.update({'a': list(highlight_props['x']), 'b': list(highlight_props['y']), 'c': list(highlight_props['z']), 'name': 'Highlighted', 'hovertext': highlight_props['texts'], 'marker': {'color': 'mediumvioletred', 'line': {'width': 2.0, 'color': 'black'}, 'symbol': 'square', 'size': 16}})
        elif self._dim == 3 and self.ternary_style == '3d':
            stable_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
            unstable_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
            stable_markers.update({'x': list(stable_props['y']), 'y': list(stable_props['x']), 'z': list(stable_props['z']), 'name': 'Stable', 'marker': {'color': '#1e1e1f', 'size': 11, 'opacity': 0.99}, 'hovertext': stable_props['texts'], 'error_z': {'array': list(stable_props['uncertainties']), 'type': 'data', 'color': 'darkgray', 'width': 10, 'thickness': 5}})
            unstable_markers.update({'x': unstable_props['y'], 'y': unstable_props['x'], 'z': unstable_props['z'], 'name': 'Above Hull', 'hovertext': unstable_props['texts'], 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 5, 'line': {'color': 'black', 'width': 1}, 'symbol': 'diamond', 'opacity': 0.7, 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}})
            if highlight_entries:
                highlight_markers = plotly_layouts['default_ternary_3d_marker_settings'].copy()
                highlight_markers.update({'x': list(highlight_props['y']), 'y': list(highlight_props['x']), 'z': list(highlight_props['z']), 'name': 'Highlighted', 'marker': {'size': 12, 'opacity': 0.99, 'symbol': 'square', 'color': 'mediumvioletred'}, 'hovertext': highlight_props['texts'], 'error_z': {'array': list(highlight_props['uncertainties']), 'type': 'data', 'color': 'darkgray', 'width': 10, 'thickness': 5}})
        elif self._dim == 4:
            stable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
            unstable_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
            stable_markers.update({'x': stable_props['x'], 'y': stable_props['y'], 'z': stable_props['z'], 'name': 'Stable', 'marker': {'size': 7, 'opacity': 0.99, 'color': 'darkgreen', 'line': {'color': 'black', 'width': 1}}, 'hovertext': stable_props['texts']})
            unstable_markers.update({'x': unstable_props['x'], 'y': unstable_props['y'], 'z': unstable_props['z'], 'name': 'Above Hull', 'marker': {'color': unstable_props['energies'], 'colorscale': plotly_layouts['unstable_colorscale'], 'size': 5, 'symbol': 'diamond', 'line': {'color': 'black', 'width': 1}, 'colorbar': {'title': 'Energy Above Hull<br>(eV/atom)', 'x': 0, 'y': 1, 'yanchor': 'top', 'xpad': 0, 'ypad': 0, 'thickness': 0.02, 'thicknessmode': 'fraction', 'len': 0.5}}, 'hovertext': unstable_props['texts'], 'visible': 'legendonly'})
            if highlight_entries:
                highlight_markers = plotly_layouts['default_quaternary_marker_settings'].copy()
                highlight_markers.update({'x': highlight_props['x'], 'y': highlight_props['y'], 'z': highlight_props['z'], 'name': 'Highlighted', 'marker': {'size': 9, 'opacity': 0.99, 'symbol': 'square', 'color': 'mediumvioletred', 'line': {'color': 'black', 'width': 1}}, 'hovertext': highlight_props['texts']})
        highlight_marker_plot = None
        if self._dim in [1, 2]:
            stable_marker_plot, unstable_marker_plot = (go.Scatter(**markers) for markers in [stable_markers, unstable_markers])
            if highlight_entries:
                highlight_marker_plot = go.Scatter(**highlight_markers)
        elif self._dim == 3 and self.ternary_style == '2d':
            stable_marker_plot, unstable_marker_plot = (go.Scatterternary(**markers) for markers in [stable_markers, unstable_markers])
            if highlight_entries:
                highlight_marker_plot = go.Scatterternary(**highlight_markers)
        else:
            stable_marker_plot, unstable_marker_plot = (go.Scatter3d(**markers) for markers in [stable_markers, unstable_markers])
            if highlight_entries:
                highlight_marker_plot = go.Scatter3d(**highlight_markers)
        return (stable_marker_plot, unstable_marker_plot, highlight_marker_plot)

    def _create_plotly_uncertainty_shading(self, stable_marker_plot):
        """
        Creates shaded uncertainty region for stable entries. Currently only works
        for binary (dim=2) phase diagrams.

        Args:
            stable_marker_plot: go.Scatter object with stable markers and their
            error bars.

        Returns:
            Plotly go.Scatter object with uncertainty window shading.
        """
        uncertainty_plot = None
        x = stable_marker_plot.x
        y = stable_marker_plot.y
        transformed = False
        if hasattr(self._pd, 'original_entries') or hasattr(self._pd, 'chempots'):
            transformed = True
        if self._dim == 2:
            error = stable_marker_plot.error_y['array']
            points = np.append(x, [y, error]).reshape(3, -1).T
            points = points[points[:, 0].argsort()]
            outline = points[:, :2].copy()
            outline[:, 1] = outline[:, 1] + points[:, 2]
            last = -1
            if transformed:
                last = None
            flipped_points = np.flip(points[:last, :].copy(), axis=0)
            flipped_points[:, 1] = flipped_points[:, 1] - flipped_points[:, 2]
            outline = np.vstack((outline, flipped_points[:, :2]))
            uncertainty_plot = go.Scatter(x=outline[:, 0], y=outline[:, 1], name='Uncertainty (window)', fill='toself', mode='lines', line={'width': 0}, fillcolor='lightblue', hoverinfo='skip', opacity=0.4)
        return uncertainty_plot

    def _create_plotly_ternary_support_lines(self):
        """
        Creates support lines which aid in seeing the ternary hull in three
        dimensions.

        Returns:
            go.Scatter3d plot of support lines for ternary phase diagram.
        """
        stable_entry_coords = dict(map(reversed, self.pd_plot_data[1].items()))
        elem_coords = [stable_entry_coords[entry] for entry in self._pd.el_refs.values()]
        x, y, z = ([], [], [])
        for line in itertools.combinations(elem_coords, 2):
            x.extend([line[0][0], line[1][0], None] * 2)
            y.extend([line[0][1], line[1][1], None] * 2)
            z.extend([0, 0, None, self._min_energy, self._min_energy, None])
        for elem in elem_coords:
            x.extend([elem[0], elem[0], None])
            y.extend([elem[1], elem[1], None])
            z.extend([0, self._min_energy, None])
        return go.Scatter3d(x=list(y), y=list(x), z=list(z), mode='lines', hoverinfo='none', line={'color': 'rgba (0, 0, 0, 0.4)', 'dash': 'solid', 'width': 1.0}, showlegend=False)

    @no_type_check
    def _get_matplotlib_2d_plot(self, label_stable=True, label_unstable=True, ordering=None, energy_colormap=None, vmin_mev=-60.0, vmax_mev=60.0, show_colorbar=True, process_attributes=False, ax: plt.Axes=None):
        """
        Shows the plot using matplotlib.

        Imports are done within the function as matplotlib is no longer the default.
        """
        ax = ax or pretty_plot(8, 6)
        if ordering is None:
            lines, labels, unstable = self.pd_plot_data
        else:
            _lines, _labels, _unstable = self.pd_plot_data
            lines, labels, unstable = order_phase_diagram(_lines, _labels, _unstable, ordering)
        if energy_colormap is None:
            if process_attributes:
                for x, y in lines:
                    plt.plot(x, y, 'k-', linewidth=3, markeredgecolor='k')
                for x, y in labels:
                    if labels[x, y].attribute is None or labels[x, y].attribute == 'existing':
                        plt.plot(x, y, 'ko', **self.plotkwargs)
                    else:
                        plt.plot(x, y, 'k*', **self.plotkwargs)
            else:
                for x, y in lines:
                    plt.plot(x, y, 'ko-', **self.plotkwargs)
        else:
            for x, y in lines:
                plt.plot(x, y, 'k-', markeredgecolor='k')
            vmin = vmin_mev / 1000.0
            vmax = vmax_mev / 1000.0
            if energy_colormap == 'default':
                mid = -vmin / (vmax - vmin)
                cmap = LinearSegmentedColormap.from_list('custom_colormap', [(0.0, '#005500'), (mid, '#55FF55'), (mid, '#FFAAAA'), (1.0, '#FF0000')])
            else:
                cmap = energy_colormap
            norm = Normalize(vmin=vmin, vmax=vmax)
            _map = ScalarMappable(norm=norm, cmap=cmap)
            _energies = [self._pd.get_equilibrium_reaction_energy(entry) for coord, entry in labels.items()]
            energies = [en if en < 0 else -1e-08 for en in _energies]
            vals_stable = _map.to_rgba(energies)
            ii = 0
            if process_attributes:
                for x, y in labels:
                    if labels[x, y].attribute is None or labels[x, y].attribute == 'existing':
                        plt.plot(x, y, 'o', markerfacecolor=vals_stable[ii], markersize=12)
                    else:
                        plt.plot(x, y, '*', markerfacecolor=vals_stable[ii], markersize=18)
                    ii += 1
            else:
                for x, y in labels:
                    plt.plot(x, y, 'o', markerfacecolor=vals_stable[ii], markersize=15)
                    ii += 1
        font = FontProperties()
        font.set_weight('bold')
        font.set_size(24)
        if len(self._pd.elements) == 3:
            plt.axis('equal')
            plt.xlim((-0.1, 1.2))
            plt.ylim((-0.1, 1.0))
            plt.axis('off')
            center = (0.5, math.sqrt(3) / 6)
        else:
            miny = min((c[1] for c in labels))
            ybuffer = max(abs(miny) * 0.1, 0.1)
            plt.xlim((-0.1, 1.1))
            plt.ylim((miny - ybuffer, ybuffer))
            center = (0.5, miny / 2)
            plt.xlabel('Fraction', fontsize=28, fontweight='bold')
            plt.ylabel('Formation energy (eV/atom)', fontsize=28, fontweight='bold')
        for coords in sorted(labels, key=lambda x: -x[1]):
            entry = labels[coords]
            label = entry.name
            vec = np.array(coords) - center
            vec = vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec
            valign = 'bottom' if vec[1] > 0 else 'top'
            if vec[0] < -0.01:
                halign = 'right'
            elif vec[0] > 0.01:
                halign = 'left'
            else:
                halign = 'center'
            if label_stable:
                if process_attributes and entry.attribute == 'new':
                    plt.annotate(latexify(label), coords, xytext=vec, textcoords='offset points', horizontalalignment=halign, verticalalignment=valign, fontproperties=font, color='g')
                else:
                    plt.annotate(latexify(label), coords, xytext=vec, textcoords='offset points', horizontalalignment=halign, verticalalignment=valign, fontproperties=font)
        if self.show_unstable:
            font = FontProperties()
            font.set_size(16)
            energies_unstable = [self._pd.get_e_above_hull(entry) for entry, coord in unstable.items()]
            if energy_colormap is not None:
                energies.extend(energies_unstable)
                vals_unstable = _map.to_rgba(energies_unstable)
            ii = 0
            for entry, coords in unstable.items():
                ehull = self._pd.get_e_above_hull(entry)
                if ehull < self.show_unstable:
                    vec = np.array(coords) - center
                    vec = vec / np.linalg.norm(vec) * 10 if np.linalg.norm(vec) != 0 else vec
                    label = entry.name
                    if energy_colormap is None:
                        plt.plot(coords[0], coords[1], 'ks', linewidth=3, markeredgecolor='k', markerfacecolor='r', markersize=8)
                    else:
                        plt.plot(coords[0], coords[1], 's', linewidth=3, markeredgecolor='k', markerfacecolor=vals_unstable[ii], markersize=8)
                    if label_unstable:
                        plt.annotate(latexify(label), coords, xytext=vec, textcoords='offset points', horizontalalignment=halign, color='b', verticalalignment=valign, fontproperties=font)
                    ii += 1
        if energy_colormap is not None and show_colorbar:
            _map.set_array(energies)
            cbar = plt.colorbar(_map)
            cbar.set_label('Energy [meV/at] above hull (positive values)\nInverse energy [meV/at] above hull (negative values)', rotation=-90, ha='center', va='bottom')
        fig = plt.gcf()
        fig.set_size_inches((8, 6))
        plt.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.07)
        return ax

    @no_type_check
    def _get_matplotlib_3d_plot(self, label_stable=True, ax: plt.Axes=None):
        """
        Shows the plot using matplotlib.

        Args:
            label_stable (bool): Whether to label stable compounds.
            ax (plt.Axes): An existing axes object (optional). If not provided, a new one will be created.

        Returns:
            plt.Axes: The axes object with the plot.
        """
        ax = ax or plt.figure().add_subplot(111, projection='3d')
        font = FontProperties(weight='bold', size=13)
        lines, labels, _ = self.pd_plot_data
        count = 1
        newlabels = []
        for x, y, z in lines:
            ax.plot(x, y, z, 'bo-', linewidth=3, markeredgecolor='b', markerfacecolor='r', markersize=10)
        for coords in sorted(labels):
            entry = labels[coords]
            label = entry.name
            if label_stable:
                if len(entry.elements) == 1:
                    ax.text(coords[0], coords[1], coords[2], label, fontproperties=font)
                else:
                    ax.text(coords[0], coords[1], coords[2], str(count), fontsize=12)
                    newlabels.append(f'{count} : {latexify(label)}')
                    count += 1
        plt.figtext(0.01, 0.01, '\n'.join(newlabels), fontproperties=font)
        ax.axis('off')
        ax.set(xlim=(-0.1, 0.72), ylim=(0, 0.66), zlim=(0, 0.56))
        return ax