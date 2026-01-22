from . import utilities
def _base_str_(self):
    if self.free_variables is None:
        f = ''
    else:
        f = ', free_variables = %r' % self.free_variables
    if self.genus is not None:
        f += ', genus = %d' % self.genus
    return 'NonZeroDimensionalComponent(dimension = %r%s)' % (self.dimension, f)