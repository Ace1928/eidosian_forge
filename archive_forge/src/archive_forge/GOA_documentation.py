import copy
Accept a record, and a dictionary of field values.

    The format is {'field_name': set([val1, val2])}.
    If any field in the record has  a matching value, the function returns
    True. Otherwise, returns False.
    