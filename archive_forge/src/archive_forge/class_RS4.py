import abc
from types import SimpleNamespace
from rpy2.robjects.robject import RObjectMixin
import rpy2.rinterface as rinterface
from rpy2.rinterface import StrSexpVector
from rpy2.robjects import help as rhelp
from rpy2.robjects import conversion
class RS4(RObjectMixin, rinterface.SexpS4):
    """ Python representation of an R instance of class 'S4'. """

    def slotnames(self):
        """ Return the 'slots' defined for this object """
        return methods_env['slotNames'](self)

    def do_slot(self, name):
        return conversion.get_conversion().rpy2py(super(RS4, self).do_slot(name))

    def extends(self):
        """Return the R classes this extends.

        This calls the R function methods::extends()."""
        return methods_env['extends'](self.rclass)

    @staticmethod
    def isclass(name):
        """ Return whether the given name is a defined class. """
        name = conversion.get_conversion().py2rpy(name)
        return methods_env['isClass'](name)[0]

    def validobject(self, test=False, complete=False):
        """ Return whether the instance is 'valid' for its class. """
        cv = conversion.get_conversion()
        test = cv.py2rpy(test)
        complete = cv.py2rpy(complete)
        return methods_env['validObject'](self, test=test, complete=complete)[0]