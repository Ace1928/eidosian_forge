import abc
from oslo_serialization import jsonutils
import six
@six.add_metaclass(abc.ABCMeta)
class CADFAbstractType(object):
    """The abstract base class for all CADF (complex) data types (classes)."""

    @abc.abstractmethod
    def is_valid(self, value):
        pass

    def as_dict(self):
        """Return dict representation of Event."""
        return jsonutils.to_primitive(self, convert_instances=True)

    def _isset(self, attr):
        """Check to see if attribute is defined."""
        try:
            if isinstance(getattr(self, attr), ValidatorDescriptor):
                return False
            return True
        except AttributeError:
            return False