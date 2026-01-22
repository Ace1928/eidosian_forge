import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
class MHPlot:
    """Makes a plot summarizing the output of the MH algorithm from the
    specified rundirectory. If no rundirectory is supplied, uses the
    current directory."""

    def __init__(self, rundirectory=None, logname='hop.log'):
        if not rundirectory:
            rundirectory = os.getcwd()
        self._rundirectory = rundirectory
        self._logname = logname
        self._read_log()
        self._fig, self._ax = self._makecanvas()
        self._plot_data()

    def get_figure(self):
        """Returns the matplotlib figure object."""
        return self._fig

    def save_figure(self, filename):
        """Saves the file to the specified path, with any allowed
        matplotlib extension (e.g., .pdf, .png, etc.)."""
        self._fig.savefig(filename)

    def _read_log(self):
        """Reads relevant parts of the log file."""
        data = []
        with open(os.path.join(self._rundirectory, self._logname), 'r') as fd:
            lines = fd.read().splitlines()
        step_almost_over = False
        step_over = False
        for line in lines:
            if line.startswith('msg: Molecular dynamics:'):
                status = 'performing MD'
            elif line.startswith('msg: Optimization:'):
                status = 'performing QN'
            elif line.startswith('ene:'):
                status = 'local optimum reached'
                energy = floatornan(line.split()[1])
            elif line.startswith('msg: Accepted new minimum.'):
                status = 'accepted'
                step_almost_over = True
            elif line.startswith('msg: Found previously found minimum.'):
                status = 'previously found minimum'
                step_almost_over = True
            elif line.startswith('msg: Re-found last minimum.'):
                status = 'previous minimum'
                step_almost_over = True
            elif line.startswith('msg: Rejected new minimum'):
                status = 'rejected'
                step_almost_over = True
            elif line.startswith('par: '):
                temperature = floatornan(line.split()[1])
                ediff = floatornan(line.split()[2])
                if step_almost_over:
                    step_over = True
                    step_almost_over = False
            if step_over:
                data.append([energy, status, temperature, ediff])
                step_over = False
        if data[-1][1] != status:
            data.append([np.nan, status, temperature, ediff])
        self._data = data

    def _makecanvas(self):
        from matplotlib import pyplot
        from matplotlib.ticker import ScalarFormatter
        fig = pyplot.figure(figsize=(6.0, 8.0))
        lm, rm, bm, tm = (0.22, 0.02, 0.05, 0.04)
        vg1 = 0.01
        vg2 = 0.03
        ratio = 2.0
        figwidth = 1.0 - lm - rm
        totalfigheight = 1.0 - bm - tm - vg1 - 2.0 * vg2
        parfigheight = totalfigheight / (2.0 * ratio + 2)
        epotheight = ratio * parfigheight
        ax1 = fig.add_axes((lm, bm, figwidth, epotheight))
        ax2 = fig.add_axes((lm, bm + epotheight + vg1, figwidth, epotheight))
        for ax in [ax1, ax2]:
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ediffax = fig.add_axes((lm, bm + 2.0 * epotheight + vg1 + vg2, figwidth, parfigheight))
        tempax = fig.add_axes((lm, bm + 2 * epotheight + vg1 + 2 * vg2 + parfigheight, figwidth, parfigheight))
        for ax in [ax2, tempax, ediffax]:
            ax.set_xticklabels([])
        ax1.set_xlabel('step')
        tempax.set_ylabel('$T$, K')
        ediffax.set_ylabel('$E_\\mathrm{diff}$, eV')
        for ax in [ax1, ax2]:
            ax.set_ylabel('$E_\\mathrm{pot}$, eV')
        ax = CombinedAxis(ax1, ax2, tempax, ediffax)
        self._set_zoomed_range(ax)
        ax1.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        return (fig, ax)

    def _set_zoomed_range(self, ax):
        """Try to intelligently set the range for the zoomed-in part of the
        graph."""
        energies = [line[0] for line in self._data if not np.isnan(line[0])]
        dr = max(energies) - min(energies)
        if dr == 0.0:
            dr = 1.0
        ax.set_ax1_range((min(energies) - 0.2 * dr, max(energies) + 0.2 * dr))

    def _plot_data(self):
        for step, line in enumerate(self._data):
            self._plot_energy(step, line)
            self._plot_qn(step, line)
            self._plot_md(step, line)
        self._plot_parameters()
        self._ax.set_xlim(self._ax.ax1.get_xlim())

    def _plot_energy(self, step, line):
        """Plots energy and annotation for acceptance."""
        energy, status = (line[0], line[1])
        if np.isnan(energy):
            return
        self._ax.plot([step, step + 0.5], [energy] * 2, '-', color='k', linewidth=2.0)
        if status == 'accepted':
            self._ax.text(step + 0.51, energy, '$\\checkmark$')
        elif status == 'rejected':
            self._ax.text(step + 0.51, energy, '$\\Uparrow$', color='red')
        elif status == 'previously found minimum':
            self._ax.text(step + 0.51, energy, '$\\hookleftarrow$', color='red', va='center')
        elif status == 'previous minimum':
            self._ax.text(step + 0.51, energy, '$\\leftarrow$', color='red', va='center')

    def _plot_md(self, step, line):
        """Adds a curved plot of molecular dynamics trajectory."""
        if step == 0:
            return
        energies = [self._data[step - 1][0]]
        file = os.path.join(self._rundirectory, 'md%05i.traj' % step)
        with io.Trajectory(file, 'r') as traj:
            for atoms in traj:
                energies.append(atoms.get_potential_energy())
        xi = step - 1 + 0.5
        if len(energies) > 2:
            xf = xi + (step + 0.25 - xi) * len(energies) / (len(energies) - 2.0)
        else:
            xf = step
        if xf > step + 0.75:
            xf = step
        self._ax.plot(np.linspace(xi, xf, num=len(energies)), energies, '-k')

    def _plot_qn(self, index, line):
        """Plots a dashed vertical line for the optimization."""
        if line[1] == 'performing MD':
            return
        file = os.path.join(self._rundirectory, 'qn%05i.traj' % index)
        if os.path.getsize(file) == 0:
            return
        with io.Trajectory(file, 'r') as traj:
            energies = [traj[0].get_potential_energy(), traj[-1].get_potential_energy()]
        if index > 0:
            file = os.path.join(self._rundirectory, 'md%05i.traj' % index)
            atoms = io.read(file, index=-3)
            energies[0] = atoms.get_potential_energy()
        self._ax.plot([index + 0.25] * 2, energies, ':k')

    def _plot_parameters(self):
        """Adds a plot of temperature and Ediff to the plot."""
        steps, Ts, ediffs = ([], [], [])
        for step, line in enumerate(self._data):
            steps.extend([step + 0.5, step + 1.5])
            Ts.extend([line[2]] * 2)
            ediffs.extend([line[3]] * 2)
        self._ax.tempax.plot(steps, Ts)
        self._ax.ediffax.plot(steps, ediffs)
        for ax in [self._ax.tempax, self._ax.ediffax]:
            ylim = ax.get_ylim()
            yrange = ylim[1] - ylim[0]
            ax.set_ylim((ylim[0] - 0.1 * yrange, ylim[1] + 0.1 * yrange))