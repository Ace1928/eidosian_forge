from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
Create a header instance for header 'name' from 'value'.

        Creates a header instance by creating a specialized class for parsing
        and representing the specified header by combining the factory
        base_class with a specialized class from the registry or the
        default_class, and passing the name and value to the constructed
        class's constructor.

        