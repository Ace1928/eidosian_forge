import re
def _create_printers():
    printer = gdb.printing.RegexpCollectionPrettyPrinter('Numba')
    printer.add_printer('Numba unaligned array printer', '^unaligned array\\(', NumbaArrayPrinter)
    printer.add_printer('Numba array printer', '^array\\(', NumbaArrayPrinter)
    printer.add_printer('Numba complex printer', '^complex[0-9]+\\ ', NumbaComplexPrinter)
    printer.add_printer('Numba Tuple printer', '^Tuple\\(', NumbaTuplePrinter)
    printer.add_printer('Numba UniTuple printer', '^UniTuple\\(', NumbaUniTuplePrinter)
    printer.add_printer('Numba unicode_type printer', '^unicode_type\\s+\\(', NumbaUnicodeTypePrinter)
    return printer