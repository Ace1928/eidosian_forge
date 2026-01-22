from oslo_versionedobjects import fields as obj_fields
def add_standard_attributes(cls):
    """Add standard attributes to the said class.

    :param cls: The class to add standard attributes to.
    :returns: None.
    """
    cls.fields = cls.fields.copy()
    cls.fields.update(STANDARD_ATTRIBUTES)
    cls.fields_no_update += ('created_at', 'updated_at')
    cls.fields_no_update.append('revision_number')