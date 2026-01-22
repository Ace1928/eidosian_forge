import os
from subprocess import Popen, PIPE
import re
import numpy as np
from ase.units import Hartree, Bohr
from ase.calculators.calculator import PropertyNotImplementedError
def _write_inp(self, atoms, fh):
    fh.write('FLEUR input generated with ASE\n')
    fh.write('\n')
    if atoms.pbc[2]:
        film = 'f'
    else:
        film = 't'
    fh.write('&input film=%s /' % film)
    fh.write('\n')
    for vec in atoms.get_cell():
        fh.write(' ')
        for el in vec:
            fh.write(' %21.16f' % (el / Bohr))
        fh.write('\n')
    fh.write(' %21.16f\n' % 1.0)
    fh.write(' %21.16f %21.16f %21.16f\n' % (1.0, 1.0, 1.0))
    fh.write('\n')
    natoms = len(atoms)
    fh.write(' %6d\n' % natoms)
    positions = atoms.get_scaled_positions()
    if not atoms.pbc[2]:
        cart_pos = atoms.get_positions()
        cart_pos[:, 2] -= atoms.get_cell()[2, 2] / 2.0
        positions[:, 2] = cart_pos[:, 2] / Bohr
    atomic_numbers = atoms.get_atomic_numbers()
    for n, (Z, pos) in enumerate(zip(atomic_numbers, positions)):
        if self.equivatoms:
            fh.write('%3d' % Z)
        else:
            fh.write('%3d.%04d' % (Z, n))
        for el in pos:
            fh.write(' %21.16f' % el)
        fh.write('\n')
    fh.write('&end /')
    try:
        inpgen = os.environ['FLEUR_INPGEN']
    except KeyError:
        raise RuntimeError('Please set FLEUR_INPGEN')
    if os.path.isfile('inp'):
        os.rename('inp', 'inp.bak')
    os.system('%s -old < inp_simple' % inpgen)
    with open('inp', 'r') as fh:
        lines = fh.readlines()
    window_ln = -1
    for ln, line in enumerate(lines):
        if line.startswith('pbe'):
            if self.xc == 'PBE':
                pass
            elif self.xc == 'RPBE':
                lines[ln] = 'rpbe   non-relativi\n'
            elif self.xc == 'LDA':
                lines[ln] = 'mjw    non-relativic\n'
                del lines[ln + 1]
            else:
                raise RuntimeError('XC-functional %s is not supported' % self.xc)
        if line.startswith('Window'):
            window_ln = ln
        if self.kmax and ln == window_ln:
            line = '%10.5f\n' % self.kmax
            lines[ln + 2] = line
        if self.lenergy is not None and ln == window_ln:
            l0 = lines[ln + 1].split()[0]
            l = lines[ln + 1].replace(l0, '%8.5f' % (self.lenergy / Hartree))
            lines[ln + 1] = l
        if self.kmax and line.startswith('vchk'):
            gmax = 3.0 * self.kmax
            line = ' %10.6f %10.6f\n' % (gmax, int(2.5 * self.kmax * 10) / 10.0)
            lines[ln - 1] = line
        if self.width and line.startswith('gauss'):
            line = 'gauss=F   %7.5ftria=F\n' % (self.width / Hartree)
            lines[ln] = line
        if self.kpts and line.startswith('nkpt'):
            line = 'nkpt=      nx=%2d,ny=%2d,nz=%2d\n' % (self.kpts[0], self.kpts[1], self.kpts[2])
            lines[ln] = line
        if self.itmax < self.itmax_step_default and line.startswith('itmax'):
            lsplit = line.split(',')
            if lsplit[0].find('itmax') != -1:
                lsplit[0] = 'itmax=' + '%2d' % self.itmax
                lines[ln] = ','.join(lsplit)
        if self.mixer and line.startswith('itmax'):
            imix = self.mixer['imix']
            alpha = self.mixer['alpha']
            spinf = self.mixer['spinf']
            line_end = 'imix=%2d,alpha=%6.2f,spinf=%6.2f\n' % (imix, alpha, spinf)
            line = line[:21] + line_end
            lines[ln] = line
        if atoms.get_initial_magnetic_moments().sum() > 0.0:
            assert not self.equivatoms, 'equivatoms currently not allowed in magnetic systems'
            if line.find('jspins=1') != -1:
                lines[ln] = line.replace('jspins=1', 'jspins=2')
            if line.startswith('swsp=F'):
                lines[ln] = 'swsp=F'
                for m in atoms.get_initial_magnetic_moments():
                    lines[ln] += ' %5.2f' % m
                lines[ln] += '\n'
        if line.startswith(' J  53'):
            lines[ln] = lines[ln].replace(' J  53', ' I  53')
    if self.rmt is not None:
        for s in list(set(atoms.get_chemical_symbols())):
            if s in self.rmt:
                for ln, line in enumerate(lines):
                    ls = line.split()
                    if len(ls) == 7 and ls[0].strip() == s:
                        rorig = ls[5].strip()
                        if self.rmt[s] < 0.0:
                            r = float(rorig) + self.rmt[s] / Bohr
                        else:
                            r = self.rmt[s] / Bohr
                        print(s, rorig, r)
                        lines[ln] = lines[ln].replace(rorig, '%.6f' % r)
    with open('inp', 'w') as fh:
        for line in lines:
            fh.write(line)