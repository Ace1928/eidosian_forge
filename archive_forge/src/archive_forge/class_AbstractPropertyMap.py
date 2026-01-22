class AbstractPropertyMap:
    """Define base class, map holder of residue properties."""

    def __init__(self, property_dict, property_keys, property_list):
        """Initialize the class."""
        self.property_dict = property_dict
        self.property_keys = property_keys
        self.property_list = property_list

    def _translate_id(self, entity_id):
        """Return entity identifier (PRIVATE)."""
        return entity_id

    def __contains__(self, id):
        """Check if the mapping has a property for this residue.

        :param chain_id: chain id
        :type chain_id: char

        :param res_id: residue id
        :type res_id: char

        Examples
        --------
        This is an incomplete but illustrative example::

            if (chain_id, res_id) in apmap:
                res, prop = apmap[(chain_id, res_id)]

        """
        translated_id = self._translate_id(id)
        return translated_id in self.property_dict

    def __getitem__(self, key):
        """Return property for a residue.

        :param chain_id: chain id
        :type chain_id: char

        :param res_id: residue id
        :type res_id: int or (char, int, char)

        :return: some residue property
        :rtype: anything (can be a tuple)
        """
        translated_id = self._translate_id(key)
        return self.property_dict[translated_id]

    def __len__(self):
        """Return number of residues for which the property is available.

        :return: number of residues
        :rtype: int
        """
        return len(self.property_dict)

    def keys(self):
        """Return the list of residues.

        :return: list of residues for which the property was calculated
        :rtype: [(chain_id, res_id), (chain_id, res_id),...]
        """
        return self.property_keys

    def __iter__(self):
        """Iterate over the (entity, property) list.

        Handy alternative to the dictionary-like access.

        :return: iterator

        Examples
        --------
        >>> entity_property_list = [
        ...     ('entity_1', 'property_1'),
        ...     ('entity_2', 'property_2')
        ... ]
        >>> map = AbstractPropertyMap({}, [], entity_property_list)
        >>> for (res, property) in iter(map):
        ...     print(res, property)
        entity_1 property_1
        entity_2 property_2

        """
        for i in range(len(self.property_list)):
            yield self.property_list[i]