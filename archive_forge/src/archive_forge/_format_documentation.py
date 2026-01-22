import datetime
import re
import socket
from jsonschema.compat import str_types
from jsonschema.exceptions import FormatError

        Check whether the instance conforms to the given format.

        Arguments:

            instance (any primitive type, i.e. str, number, bool):

                The instance to check

            format (str):

                The format that instance should conform to

        Returns:

            bool: Whether it conformed

        