import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def _write_potential(self, fd, nc, numformat):
    assert self.nr % nc == 0
    assert self.nrho % nc == 0
    for line in self.header:
        fd.write(line)
    fd.write('{0} '.format(self.Nelements).encode())
    fd.write(' '.join(self.elements).encode() + b'\n')
    fd.write(('%d %f %d %f %f \n' % (self.nrho, self.drho, self.nr, self.dr, self.cutoff)).encode())
    rs = np.arange(0, self.nr) * self.dr
    rhos = np.arange(0, self.nrho) * self.drho
    for i in range(self.Nelements):
        fd.write(('%d %f %f %s\n' % (self.Z[i], self.mass[i], self.a[i], str(self.lattice[i]))).encode())
        np.savetxt(fd, self.embedded_energy[i](rhos).reshape(self.nrho // nc, nc), fmt=nc * [numformat])
        if self.form == 'fs':
            for j in range(self.Nelements):
                np.savetxt(fd, self.electron_density[i, j](rs).reshape(self.nr // nc, nc), fmt=nc * [numformat])
        else:
            np.savetxt(fd, self.electron_density[i](rs).reshape(self.nr // nc, nc), fmt=nc * [numformat])
    for i in range(self.Nelements):
        for j in range(i, self.Nelements):
            np.savetxt(fd, (rs * self.phi[i, j](rs)).reshape(self.nr // nc, nc), fmt=nc * [numformat])
    if self.form == 'adp':
        for i in range(self.Nelements):
            for j in range(i + 1):
                np.savetxt(fd, self.d_data[i, j])
        for i in range(self.Nelements):
            for j in range(i + 1):
                np.savetxt(fd, self.q_data[i, j])