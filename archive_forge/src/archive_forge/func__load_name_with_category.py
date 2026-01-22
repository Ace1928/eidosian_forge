import logging
from botocore import xform_name
def _load_name_with_category(self, names, name, category, snake_case=True):
    """
        Load a name with a given category, possibly renaming it
        if that name is already in use. The name will be stored
        in ``names`` and possibly be set up in ``self._renamed``.

        :type names: set
        :param names: Existing names (Python attributes, properties, or
                      methods) on the resource.
        :type name: string
        :param name: The original name of the value.
        :type category: string
        :param category: The value type, such as 'identifier' or 'action'
        :type snake_case: bool
        :param snake_case: True (default) if the name should be snake cased.
        """
    if snake_case:
        name = xform_name(name)
    if name in names:
        logger.debug('Renaming %s %s %s' % (self.name, category, name))
        self._renamed[category, name] = name + '_' + category
        name += '_' + category
        if name in names:
            raise ValueError('Problem renaming {0} {1} to {2}!'.format(self.name, category, name))
    names.add(name)