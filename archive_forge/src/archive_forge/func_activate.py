import warnings
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
import rpy2.robjects.conversion as conversion
from rpy2.robjects.packages import importr, WeakPackage
def activate():
    warnings.warn('The global conversion available with activate() is deprecated and will be removed in the next major release. Use a local converter.', category=DeprecationWarning)
    global original_converter
    if original_converter is not None:
        return
    original_converter = conversion.converter
    new_converter = conversion.Converter('grid conversion', template=original_converter)
    for k, v in py2rpy.registry.items():
        if k is object:
            continue
        new_converter.py2rpy.register(k, v)
    for k, v in rpy2py.registry.items():
        if k is object:
            continue
        new_converter.rpy2py.register(k, v)
    conversion.set_conversion(new_converter)