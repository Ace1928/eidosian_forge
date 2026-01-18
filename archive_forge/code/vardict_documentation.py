from __future__ import absolute_import, division, print_function
import copy
Set the value and optionally metadata for a variable. The variable is not required to exist prior to calling `set`.

        For details on the accepted metada see the documentation for method `set_meta`.

        Args:
            name (str): name of the variable being changed
            value (any): the value of the variable, it can be of any type

        Raises:
            ValueError: Raised if trying to set a variable with a reserved name.
        