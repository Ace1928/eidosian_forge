from types import ModuleType
from typing import Any, Union
import modin.pandas as pd
def _set_attribute_on_obj(name: str, extensions_dict: dict, obj: Union[pd.DataFrame, pd.Series, ModuleType]):
    """
    Create a new or override existing attribute on obj.

    Parameters
    ----------
    name : str
        The name of the attribute to assign to `obj`.
    extensions_dict : dict
        The dictionary mapping extension name to `new_attr` (assigned below).
    obj : DataFrame, Series, or modin.pandas
        The object we are assigning the new attribute to.

    Returns
    -------
    decorator
        Returns the decorator function.
    """

    def decorator(new_attr: Any):
        """
        The decorator for a function or class to be assigned to name

        Parameters
        ----------
        new_attr : Any
            The new attribute to assign to name.

        Returns
        -------
        new_attr
            Unmodified new_attr is return from the decorator.
        """
        extensions_dict[name] = new_attr
        setattr(obj, name, new_attr)
        return new_attr
    return decorator