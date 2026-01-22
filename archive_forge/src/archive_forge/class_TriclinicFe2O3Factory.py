from ase.lattice.cubic import DiamondFactory, SimpleCubicFactory
from ase.lattice.tetragonal import SimpleTetragonalFactory
from ase.lattice.triclinic import TriclinicFactory
from ase.lattice.hexagonal import HexagonalFactory
class TriclinicFe2O3Factory(TriclinicFactory):
    """A factory for creating hematite (Fe2O3) lattices.

     Rhombohedral unit cell.
     Pauling L, Hendricks S B
     Journal of the American Chemical Society 47 (1925) 781-790

     Example::

         #!/usr/bin/env python3

         from ase.lattice.hexagonal import *
         from ase.lattice.compounds import *
         import ase.io as io
         from ase import Atoms, Atom

         index1=3
         index2=3
         index3=3
         mya = 5.42
         myb = 5.42
         myc = 5.42
         myalpha = 55.28
         mybeta = 55.28
         mygamma = 55.28
         gra = TRI_Fe2O3(symbol = ('Fe', 'O'),
                         latticeconstant={'a':mya,'b':myb, 'c':myc,
                                          'alpha':myalpha,
                                          'beta':mybeta,
                                          'gamma':mygamma},
                         size=(index1,index2,index3))
         io.write('rhombohedralUC_Fe2O3.xyz', gra, format='xyz')

     """
    bravais_basis = [[0.10534, 0.10534, 0.10534], [0.39466, 0.39466, 0.39466], [0.60534, 0.60534, 0.60534], [0.89466, 0.89466, 0.89466], [0.30569, 0.69431, 0.0], [0.69431, 0.0, 0.30569], [0.0, 0.30569, 0.69431], [0.19431, 0.80569, 0.5], [0.80569, 0.5, 0.19431], [0.5, 0.19431, 0.80569]]
    element_basis = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1)