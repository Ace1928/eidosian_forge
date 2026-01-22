import crcmod
def _get_definition_by_name(crc_name):
    definition = _crc_definitions_by_name.get(_simplify_name(crc_name), None)
    if not definition:
        definition = _crc_definitions_by_identifier.get(crc_name, None)
    if not definition:
        raise KeyError("Unkown CRC name '{0}'".format(crc_name))
    return definition