from itertools import chain
from .periodic import groups, symbols, names

    This will be working examples eventually:

    #>>> ions_from_formula('NaCl') == {'Na+': 1, 'Cl-': 1}
    #True
    #>>> ions_from_formula('Fe(NO3)3') == {'Fe+3': 1, 'NO3-': 3}
    #True
    #>>> ions_from_formula('FeSO4') == {'Fe+2': 1, 'SO4-2': 1}
    #True
    #>>> ions_from_formula('(NH4)3PO4') == {'NH4+': 3, 'PO4-3': 1}
    #True
    #>>> ions_from_formula('KAl(SO4)2.11H2O') == {'K+': 1, 'Al+3': 1, 'SO4-2': 2}
    #True

    