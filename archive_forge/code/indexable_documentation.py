from .. import inspect
from ..ext.hybrid import hybrid_property
from ..orm.attributes import flag_modified
Create a new :class:`.index_property`.

        :param attr_name:
            An attribute name of an `Indexable` typed column, or other
            attribute that returns an indexable structure.
        :param index:
            The index to be used for getting and setting this value.  This
            should be the Python-side index value for integers.
        :param default:
            A value which will be returned instead of `AttributeError`
            when there is not a value at given index.
        :param datatype: default datatype to use when the field is empty.
            By default, this is derived from the type of index used; a
            Python list for an integer index, or a Python dictionary for
            any other style of index.   For a list, the list will be
            initialized to a list of None values that is at least
            ``index`` elements long.
        :param mutable: if False, writes and deletes to the attribute will
            be disallowed.
        :param onebased: assume the SQL representation of this value is
            one-based; that is, the first index in SQL is 1, not zero.
        