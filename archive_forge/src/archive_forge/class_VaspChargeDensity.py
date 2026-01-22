import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
class VaspChargeDensity:
    """Class for representing VASP charge density.

    Filename is normally CHG."""

    def __init__(self, filename):
        self.atoms = []
        self.chg = []
        self.chgdiff = []
        self.aug = ''
        self.augdiff = ''
        if filename is not None:
            self.read(filename)

    def is_spin_polarized(self):
        if len(self.chgdiff) > 0:
            return True
        return False

    def _read_chg(self, fobj, chg, volume):
        """Read charge from file object

        Utility method for reading the actual charge density (or
        charge density difference) from a file object. On input, the
        file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The chg array must be of the correct dimensions.

        """
        for zz in range(chg.shape[2]):
            for yy in range(chg.shape[1]):
                chg[:, yy, zz] = np.fromfile(fobj, count=chg.shape[0], sep=' ')
        chg /= volume

    def read(self, filename):
        """Read CHG or CHGCAR file.

        If CHG contains charge density from multiple steps all the
        steps are read and stored in the object. By default VASP
        writes out the charge density every 10 steps.

        chgdiff is the difference between the spin up charge density
        and the spin down charge density and is thus only read for a
        spin-polarized calculation.

        aug is the PAW augmentation charges found in CHGCAR. These are
        not parsed, they are just stored as a string so that they can
        be written again to a CHGCAR format file.

        """
        import ase.io.vasp as aiv
        fd = open(filename)
        self.atoms = []
        self.chg = []
        self.chgdiff = []
        self.aug = ''
        self.augdiff = ''
        while True:
            try:
                atoms = aiv.read_vasp(fd)
            except (IOError, ValueError, IndexError):
                break
            fd.readline()
            ngr = fd.readline().split()
            ng = (int(ngr[0]), int(ngr[1]), int(ngr[2]))
            chg = np.empty(ng)
            self._read_chg(fd, chg, atoms.get_volume())
            self.chg.append(chg)
            self.atoms.append(atoms)
            fl = fd.tell()
            line1 = fd.readline()
            if line1 == '':
                break
            elif line1.find('augmentation') != -1:
                augs = [line1]
                while True:
                    line2 = fd.readline()
                    if line2.split() == ngr:
                        self.aug = ''.join(augs)
                        augs = []
                        chgdiff = np.empty(ng)
                        self._read_chg(fd, chgdiff, atoms.get_volume())
                        self.chgdiff.append(chgdiff)
                    elif line2 == '':
                        break
                    else:
                        augs.append(line2)
                if len(self.aug) == 0:
                    self.aug = ''.join(augs)
                    augs = []
                else:
                    self.augdiff = ''.join(augs)
                    augs = []
            elif line1.split() == ngr:
                chgdiff = np.empty(ng)
                self._read_chg(fd, chgdiff, atoms.get_volume())
                self.chgdiff.append(chgdiff)
            else:
                fd.seek(fl)
        fd.close()

    def _write_chg(self, fobj, chg, volume, format='chg'):
        """Write charge density

        Utility function similar to _read_chg but for writing.

        """
        chgtmp = chg.T.ravel()
        chgtmp = chgtmp * volume
        chgtmp = tuple(chgtmp)
        if format.lower() == 'chg':
            for ii in range((len(chgtmp) - 1) // 10):
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G\n' % chgtmp[ii * 10:(ii + 1) * 10])
            if len(chgtmp) % 10 == 0:
                fobj.write(' %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G %#11.5G' % chgtmp[len(chgtmp) - 10:len(chgtmp)])
            else:
                for ii in range(len(chgtmp) % 10):
                    fobj.write(' %#11.5G' % chgtmp[len(chgtmp) - len(chgtmp) % 10 + ii])
        else:
            for ii in range((len(chgtmp) - 1) // 5):
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E\n' % chgtmp[ii * 5:(ii + 1) * 5])
            if len(chgtmp) % 5 == 0:
                fobj.write(' %17.10E %17.10E %17.10E %17.10E %17.10E' % chgtmp[len(chgtmp) - 5:len(chgtmp)])
            else:
                for ii in range(len(chgtmp) % 5):
                    fobj.write(' %17.10E' % chgtmp[len(chgtmp) - len(chgtmp) % 5 + ii])
        fobj.write('\n')

    def write(self, filename, format=None):
        """Write VASP charge density in CHG format.

        filename: str
            Name of file to write to.
        format: str
            String specifying whether to write in CHGCAR or CHG
            format.

        """
        import ase.io.vasp as aiv
        if format is None:
            if filename.lower().find('chgcar') != -1:
                format = 'chgcar'
            elif filename.lower().find('chg') != -1:
                format = 'chg'
            elif len(self.chg) == 1:
                format = 'chgcar'
            else:
                format = 'chg'
        with open(filename, 'w') as fd:
            for ii, chg in enumerate(self.chg):
                if format == 'chgcar' and ii != len(self.chg) - 1:
                    continue
                aiv.write_vasp(fd, self.atoms[ii], direct=True, long_format=False)
                fd.write('\n')
                for dim in chg.shape:
                    fd.write(' %4i' % dim)
                fd.write('\n')
                vol = self.atoms[ii].get_volume()
                self._write_chg(fd, chg, vol, format)
                if format == 'chgcar':
                    fd.write(self.aug)
                if self.is_spin_polarized():
                    if format == 'chg':
                        fd.write('\n')
                    for dim in chg.shape:
                        fd.write(' %4i' % dim)
                    fd.write('\n')
                    self._write_chg(fd, self.chgdiff[ii], vol, format)
                    if format == 'chgcar':
                        fd.write(self.augdiff)
                if format == 'chg' and len(self.chg) > 1:
                    fd.write('\n')