from . import xpktools
def _col_ave(elements, col):
    total = 0.0
    for element in elements:
        total += float(element.split()[col])
    return total / len(elements)