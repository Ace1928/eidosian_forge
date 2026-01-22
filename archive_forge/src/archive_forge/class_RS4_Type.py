import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
class RS4_Type(abc.ABCMeta):

    def __new__(mcs, name, bases, cls_dict):
        try:
            cls_rname = cls_dict['__rname__']
        except KeyError:
            cls_rname = name
        try:
            accessors = cls_dict['__accessors__']
        except KeyError:
            accessors = []
        for rname, where, python_name, as_property, docstring in accessors:
            if where is None:
                where = rinterface.globalenv
            else:
                where = StrSexpVector(('package:%s' % where,))
            if python_name is None:
                python_name = rname
            signature = StrSexpVector((cls_rname,))
            r_meth = getmethod(StrSexpVector((rname,)), signature=signature, where=where)
            r_meth = conversion.get_conversion().rpy2py(r_meth)
            if as_property:
                cls_dict[python_name] = property(r_meth, None, None, doc=docstring)
            else:
                cls_dict[python_name] = lambda self: r_meth(self)
        return type.__new__(mcs, name, bases, cls_dict)