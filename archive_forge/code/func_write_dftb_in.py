import os
import numpy as np
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
from ase.units import Hartree, Bohr
def write_dftb_in(self, outfile):
    """ Write the innput file for the dftb+ calculation.
            Geometry is taken always from the file 'geo_end.gen'.
        """
    outfile.write('Geometry = GenFormat { \n')
    outfile.write('    <<< "geo_end.gen" \n')
    outfile.write('} \n')
    outfile.write(' \n')
    params = self.parameters.copy()
    s = 'Hamiltonian_MaxAngularMomentum_'
    for key in params:
        if key.startswith(s) and len(key) > len(s):
            break
    else:
        symbols = set(self.atoms.get_chemical_symbols())
        for symbol in symbols:
            path = os.path.join(self.slako_dir, '{0}-{0}.skf'.format(symbol))
            l = read_max_angular_momentum(path)
            params[s + symbol] = '"{}"'.format('spdf'[l])
    previous_key = 'dummy_'
    myspace = ' '
    for key, value in sorted(params.items()):
        current_depth = key.rstrip('_').count('_')
        previous_depth = previous_key.rstrip('_').count('_')
        for my_backsclash in reversed(range(previous_depth - current_depth)):
            outfile.write(3 * (1 + my_backsclash) * myspace + '} \n')
        outfile.write(3 * current_depth * myspace)
        if key.endswith('_') and len(value) > 0:
            outfile.write(key.rstrip('_').rsplit('_')[-1] + ' = ' + str(value) + '{ \n')
        elif key.endswith('_') and len(value) == 0 and (current_depth == 0):
            outfile.write(key.rstrip('_').rsplit('_')[-1] + ' ' + str(value) + '{ \n')
        elif key.endswith('_') and len(value) == 0 and (current_depth > 0):
            outfile.write(key.rstrip('_').rsplit('_')[-1] + ' = ' + str(value) + '{ \n')
        elif key.count('_empty') == 1:
            outfile.write(str(value) + ' \n')
        elif key == 'Hamiltonian_ReadInitialCharges' and str(value).upper() == 'YES':
            f1 = os.path.isfile(self.directory + os.sep + 'charges.dat')
            f2 = os.path.isfile(self.directory + os.sep + 'charges.bin')
            if not (f1 or f2):
                print('charges.dat or .bin not found, switching off guess')
                value = 'No'
            outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
        else:
            outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
        if self.pcpot is not None and 'DFTB' in str(value):
            outfile.write('   ElectricField = { \n')
            outfile.write('      PointCharges = { \n')
            outfile.write('         CoordsAndCharges [Angstrom] = DirectRead { \n')
            outfile.write('            Records = ' + str(len(self.pcpot.mmcharges)) + ' \n')
            outfile.write('            File = "dftb_external_charges.dat" \n')
            outfile.write('         } \n')
            outfile.write('      } \n')
            outfile.write('   } \n')
        previous_key = key
    current_depth = key.rstrip('_').count('_')
    for my_backsclash in reversed(range(current_depth)):
        outfile.write(3 * my_backsclash * myspace + '} \n')
    outfile.write('ParserOptions { \n')
    outfile.write('   IgnoreUnprocessedNodes = Yes  \n')
    outfile.write('} \n')
    if self.do_forces:
        outfile.write('Analysis { \n')
        outfile.write('   CalculateForces = Yes  \n')
        outfile.write('} \n')