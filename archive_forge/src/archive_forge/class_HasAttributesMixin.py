import sys
import h5py
import numpy as np
from . import core
class HasAttributesMixin:
    _initialized = False

    def getncattr(self, name):
        """Retrieve a netCDF4 attribute."""
        return self.attrs[name]

    def setncattr(self, name, value):
        """Set a netCDF4 attribute."""
        self.attrs[name] = value

    def ncattrs(self):
        """Return netCDF4 attribute names."""
        return list(self.attrs)

    def __getattr__(self, name):
        try:
            return self.attrs[name]
        except KeyError:
            raise AttributeError(f'NetCDF: attribute {type(self).__name__}:{name} not found')

    def __setattr__(self, name, value):
        if self._initialized and name not in self.__dict__:
            self.attrs[name] = value
        else:
            object.__setattr__(self, name, value)