class AmbiguousDeb822FieldKeyError(KeyError):
    """Specialized version of KeyError to denote a valid but ambiguous field name

    This exception occurs if:
      * the field is accessed via a str on a configured view that does not automatically
        resolve ambiguous field names (see Deb822ParagraphElement.configured_view), AND
      * a concrete paragraph contents a repeated field (which is not valid in deb822
        but the module supports parsing them)

    Note that the default is to automatically resolve ambiguous fields. Accordingly
    you will only see this exception if you have "opted in" on wanting to know that
    the lookup was ambiguous.

    The ambiguity can be resolved by using a tuple of (<field-name>, <filed-index>)
    instead of <field-name>.
    """