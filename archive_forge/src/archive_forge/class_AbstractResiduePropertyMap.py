class AbstractResiduePropertyMap(AbstractPropertyMap):
    """Define class for residue properties map."""

    def __init__(self, property_dict, property_keys, property_list):
        """Initialize the class."""
        AbstractPropertyMap.__init__(self, property_dict, property_keys, property_list)

    def _translate_id(self, ent_id):
        """Return entity identifier on residue (PRIVATE)."""
        chain_id, res_id = ent_id
        if isinstance(res_id, int):
            ent_id = (chain_id, (' ', res_id, ' '))
        return ent_id